"""Mortality forecasting: Lee-Carter (SVD + drift) and neural forecasters on log m(x,t).

Everything follows Richman, Scognamiglio & Wüthrich (2026), AI Tools for Actuaries,
Ch. 8 (the benchmark, see conf/base/parameters_benchmarks.yml): Lee-Carter with
random-walk-with-drift extrapolation (R `rwf(kt, drift = TRUE)`), and networks on
10-year windows of log-mortality across all ages that predict the next year,
forecasting recursively. Keras layers are mirrored where PyTorch differs: an LSTM with
linear activation and one bias per gate, and attention with a learned scale. Every model exposes
`fit_forecast(data, params, training) -> (results, history)`, where `results` is
the long table [Sex, Year, Age, logmx, split, pred].
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim


def prepare(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    d = raw[
        (raw.Age < params["max_age"])
        & (raw.Year >= params["first_year"])
        & (raw.Year <= params["last_year"])
    ]
    long = d.melt(
        id_vars=["Year", "Age"], value_vars=params["sexes"], var_name="Sex", value_name="mx"
    )
    long["logmx"] = np.log(np.maximum(long.mx, np.exp(-11)))  # zero rates at young ages
    long["split"] = np.where(long.Year <= params["obs_year"], "learn", "test")
    return long.sort_values(["Sex", "Year", "Age"]).reset_index(drop=True)


# --- Lee-Carter --------------------------------------------------------------------------------


def lee_carter(d: pd.DataFrame, obs_year: int) -> tuple[np.ndarray, dict]:
    learn = d[d.Year <= obs_year]
    M = learn.pivot(index="Age", columns="Year", values="logmx").to_numpy()
    ax = M.mean(axis=1)
    U, s, Vt = np.linalg.svd(M - ax[:, None], full_matrices=False)
    bx, kt = U[:, 0] * s[0], Vt[0]
    c1, c2 = kt.mean(), bx.sum()
    ax, bx, kt = ax + c1 * bx, bx / c2, (kt - c1) * c2  # identifiability: sum b = 1, sum k = 0
    drift = (kt[-1] - kt[0]) / (len(kt) - 1)  # random walk with drift: mean of k_t - k_{t-1}
    h = d[d.Year > obs_year].Year.nunique()
    k_all = np.r_[kt, kt[-1] + drift * np.arange(1, h + 1)]
    fitted = ax[:, None] + np.outer(bx, k_all)  # ages x years
    return fitted, {"ax": ax.tolist(), "bx": bx.tolist(), "kt": k_all.tolist(), "drift": drift}


# --- Neural forecasters ------------------------------------------------------------------------


class CNN(nn.Module):
    def __init__(self, lookback, n_age, filters=3, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(n_age, filters, kernel)
        self.dense = nn.Linear(filters * (lookback - kernel + 1), n_age)

    def forward(self, x):
        return self.dense(self.conv(x.permute(0, 2, 1)).flatten(1))


class CNNTanh(nn.Module):
    def __init__(self, lookback, n_age, filters=3, kernel=3, qq=(20, 20)):
        super().__init__()
        self.d1 = nn.Linear(n_age, qq[0])
        self.conv = nn.Conv1d(qq[0], filters, kernel)
        self.d2 = nn.Linear(filters * (lookback - kernel + 1), qq[1])
        self.d3 = nn.Linear(qq[1], n_age)

    def forward(self, x):
        z = torch.tanh(self.d1(x)).permute(0, 2, 1)
        return self.d3(torch.tanh(self.d2(self.conv(z).flatten(1))))


class KerasLSTM(nn.Module):
    """LSTM as Keras `layer_lstm(units, activation)`: sigmoid gates, one bias per gate,
    and `activation` on the cell input and output (tanh in PyTorch's nn.LSTM)."""

    def __init__(self, n_in, units, activation=None):
        super().__init__()
        self.units, self.act = units, activation or (lambda z: z)
        self.gates = nn.Linear(n_in + units, 4 * units)

    def forward(self, x):
        h = x.new_zeros(x.shape[0], self.units)
        c = x.new_zeros(x.shape[0], self.units)
        for t in range(x.shape[1]):
            i, f, g, o = self.gates(torch.cat([x[:, t], h], 1)).chunk(4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * self.act(g)
            h = torch.sigmoid(o) * self.act(c)
        return h


class LSTM(nn.Module):
    """Ch. 8: time-distributed dense (q units), linear-activation LSTM, dense to all ages."""

    def __init__(self, lookback, n_age, q=20, units=20):
        super().__init__()
        self.time_dense = nn.Linear(n_age, q)
        self.lstm = KerasLSTM(q, units)
        self.out = nn.Linear(units, n_age)

    def forward(self, x):
        return self.out(self.lstm(self.time_dense(x)))


class Transformer(nn.Module):
    """Ch. 8: one attention block (Keras `layer_attention(use_scale = TRUE)`, i.e. dot-product
    scores times a learned scalar), skip connections and layer normalisation."""

    def __init__(self, lookback, n_age, units=3):
        super().__init__()
        self.units = units
        self.rep = nn.Linear(n_age, units)
        self.q, self.k, self.v = (nn.Linear(units, units) for _ in range(3))
        self.scale = nn.Parameter(torch.ones(()))
        self.norm = nn.LayerNorm(units)
        self.ff = nn.Linear(units, units)
        self.out = nn.Linear(units * lookback, n_age)

    def forward(self, x):
        r = self.rep(x)
        att = torch.softmax(self.scale * self.q(r) @ self.k(r).transpose(-2, -1), dim=-1)
        s1 = att @ self.v(r) + r
        return self.out((self.ff(self.norm(s1)) + s1).flatten(1))


NETS = {"cnn": CNN, "cnn_tanh": CNNTanh, "lstm": LSTM, "transformer": Transformer}


def _windows(scaled: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """scaled: years x ages. Returns X (n, lookback, ages), y (n, ages)."""
    X = np.stack([scaled[i - lookback : i] for i in range(lookback, len(scaled))])
    return X, scaled[lookback:]


def _train(net, X, y, t: dict) -> dict:
    rng = np.random.RandomState(t.get("shuffle_seed", 100))
    idx = rng.permutation(len(X))
    X, y = torch.FloatTensor(X[idx]), torch.FloatTensor(y[idx])
    cut = int(0.9 * len(X))
    Xt, yt, Xv, yv = X[:cut], y[:cut], X[cut:], y[cut:]
    opt = optim.Adam(net.parameters(), lr=t.get("lr", 0.001))
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.9, patience=t.get("patience", 5))
    loss_fn = nn.MSELoss()
    best, best_state, hist = (
        float("inf"),
        copy.deepcopy(net.state_dict()),
        {"loss": [], "val_loss": []},
    )
    bs = t.get("batch_size", 4)
    gen = torch.Generator().manual_seed(t.get("shuffle_seed", 100))
    for _ in range(t["epochs"]):
        net.train()
        order = torch.randperm(len(Xt), generator=gen)  # Keras fit() reshuffles every epoch
        for i in range(0, len(Xt), bs):
            b = order[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl, tl = loss_fn(net(Xv), yv).item(), loss_fn(net(Xt), yt).item()
        hist["loss"].append(tl)
        hist["val_loss"].append(vl)
        sched.step(vl)
        if vl < best:
            best, best_state = vl, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["val_loss"]))
    return hist


def neural_forecast(d: pd.DataFrame, kind: str, params: dict, t: dict, obs_year: int):
    lookback = t["lookback"]
    years = np.sort(d.Year.unique())
    M = d.pivot(index="Year", columns="Age", values="logmx")
    scaler = MinMaxScaler()
    # The reference notebook fits the scaler on all years (including the test period).
    # Set training.scaler_fit: learn to avoid that look-ahead in production.
    fit_on = (
        M.to_numpy().reshape(-1, 1)
        if t.get("scaler_fit", "all") == "all"
        else M.loc[:obs_year].to_numpy().reshape(-1, 1)
    )
    scaler.fit(fit_on)
    S = scaler.transform(M.to_numpy().reshape(-1, 1)).reshape(M.shape)
    n_learn = int((years <= obs_year).sum())
    X, y = _windows(S[:n_learn], lookback)
    torch.manual_seed(params.get("seed", 100))
    net = NETS[kind](lookback, M.shape[1], **params.get("arch", {}))
    hist = _train(net, X, y, t)
    net.eval()
    pred = np.full(S.shape, np.nan)
    with torch.no_grad():
        pred[lookback:n_learn] = net(torch.FloatTensor(X)).numpy()
        window = S[n_learn - lookback : n_learn].copy()
        for i in range(n_learn, len(years)):  # recursive multi-step forecast
            nxt = net(torch.FloatTensor(window[None])).numpy()[0]
            pred[i] = nxt
            window = np.vstack([window[1:], nxt])
    out = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(S.shape)
    return out.T, hist  # ages x years


def fit_forecast(data: pd.DataFrame, params: dict, training: dict, obs_year: int):
    """Fit per sex; return the long results table and the training history."""
    parts, hist = [], {}
    for sex, d in data.groupby("Sex"):
        d = d.sort_values(["Year", "Age"])
        if params["kind"] == "lee_carter":
            fitted, h = lee_carter(d, obs_year)
        else:
            fitted, h = neural_forecast(d, params["kind"], params, training, obs_year)
        years = np.sort(d.Year.unique())
        ages = np.sort(d.Age.unique())
        f = pd.DataFrame(fitted, index=ages, columns=years).stack().rename("pred")
        f.index.names = ["Age", "Year"]
        parts.append(d.merge(f.reset_index(), on=["Age", "Year"]))
        hist[sex] = h
    return pd.concat(parts, ignore_index=True), hist
