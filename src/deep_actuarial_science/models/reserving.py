"""Claims reserving on CAS Schedule P paid triangles (10 accident x 10 development years).

The CAS data include the realised lower triangle, so every model is back-tested
against what was actually paid. Models predict incremental payments for the lower
triangle; `predictions` is the long table with a `pred` column for those cells.

- chain_ladder: volume-weighted chain-ladder per company (= ODP GLM, Renshaw & Verrall 1998)
- neural_odp: CANN in the spirit of Gabrielli, Richman & Wüthrich (2020). The chain-ladder
  mean is the offset; a network with company, line and development embeddings learns a
  multiplicative correction, initialised at zero, trained jointly across companies. The
  number of epochs is chosen out of time: offsets and training cells use calendar years up
  to 1996 only, and the 1997 diagonal is the validation set. The network is then refitted on
  the whole upper triangle (offsets from all of it) for that many epochs. Validating on the
  1997 diagonal with offsets that were themselves fitted to it would make any learning look
  worse than the offset alone, so the best epoch would always be the first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from deep_actuarial_science.models.torch_utils import device

LAST_YEAR = 1997
KEY = ["lob", "company"]


def prepare(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    d = raw[raw.lob.isin(params["lines"])].rename(
        columns={
            "GRCODE": "company",
            "AccidentYear": "ay",
            "DevelopmentLag": "dev",
            "CumPaidLoss": "cum_paid",
            "EarnedPremNet": "premium",
        }
    )
    d = d[["lob", "company", "GRNAME", "ay", "dev", "cum_paid", "premium"]]
    d["cum_paid"] = d.cum_paid.clip(lower=1)  # Meyers (2015): cpdloss = pmax(1, cpdloss)
    listed = params.get("companies")
    if listed:  # a published company list, e.g. Meyers (2015), Table 12
        keep = pd.DataFrame([(lob, c) for lob, codes in listed.items() for c in codes], columns=KEY)
        d = d.merge(keep).sort_values(KEY + ["ay", "dev"])
        return _finish(d)
    g = d.groupby(KEY)
    ok = (
        (g.size() == 100)
        & g.premium.min().gt(0)
        & d[d.dev == 1].groupby(KEY).cum_paid.min().gt(0)
        & d[d.dev == 10].groupby(KEY).cum_paid.min().gt(0)
        & d.groupby(KEY).apply(
            lambda x: (
                (x.sort_values("dev").groupby("ay").cum_paid.diff().fillna(0) >= 0).mean()
                >= params.get("min_monotone", 0.95)
            ),
            include_groups=False,
        )
    )
    keep = ok[ok].index.to_frame(index=False)
    size = d.merge(keep).groupby(KEY).premium.sum().sort_values(ascending=False)
    if params.get("max_companies"):
        size = size.groupby(level="lob", group_keys=False).head(params["max_companies"])
    d = d.merge(size.index.to_frame(index=False)).sort_values(KEY + ["ay", "dev"])
    return _finish(d)


def _finish(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["incr_paid"] = d.groupby(KEY + ["ay"]).cum_paid.diff().fillna(d.cum_paid)
    d["calendar"] = d.ay + d.dev - 1
    d["upper"] = d.calendar <= LAST_YEAR
    return d.reset_index(drop=True)


# --- Chain-ladder ------------------------------------------------------------------------------


def chain_ladder_factors(tri: pd.DataFrame) -> np.ndarray:
    """Volume-weighted age-to-age factors f_1..f_9 from the upper triangle of one company."""
    C = tri[tri.upper].pivot(index="ay", columns="dev", values="cum_paid")
    C = C.reindex(columns=range(1, 11))
    f = []
    for j in range(1, 10):
        rows = C[j + 1].notna()  # no observation of development j+1 yet: no development (1.0)
        f.append(C.loc[rows, j + 1].sum() / C.loc[rows, j].sum() if rows.any() else 1.0)
    return np.array(f)


def _latest(up: pd.DataFrame) -> pd.DataFrame:
    """Latest diagonal of an upper triangle: cumulative paid and development per accident year."""
    return up.sort_values("dev").groupby("ay").last()[["dev", "cum_paid"]]


def chain_ladder(tri: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Project each accident year from its latest diagonal with volume-weighted factors."""
    parts, factors = [], {}
    for (lob, comp), t in tri.groupby(KEY):
        f = chain_ladder_factors(t)
        incr = {}
        for ay, row in _latest(t[t.upper]).iterrows():
            c = row.cum_paid
            for j in range(int(row.dev) + 1, 11):
                c_next = c * f[j - 2]  # factor from development j-1 to j
                incr[(ay, j)] = c_next - c
                c = c_next
        parts.append(t.assign(pred=[incr.get((a, j), np.nan) for a, j in zip(t.ay, t.dev)]))
        factors[f"{lob}-{comp}"] = f.tolist()
    return pd.concat(parts, ignore_index=True), {"factors": factors}


def chain_ladder_fitted_upper(tri: pd.DataFrame) -> pd.Series:
    """ODP/chain-ladder fitted means for the upper triangle (row and column sums match)."""
    out = []
    for _, t in tri.groupby(KEY):
        f = chain_ladder_factors(t)
        cum_f = np.r_[np.cumprod(f[::-1])[::-1], 1.0]  # f_j * ... * f_9, development j=1..10
        beta = np.r_[1 / cum_f[0], np.diff(1 / cum_f)]  # payment pattern
        up = t[t.upper]
        latest = _latest(up)
        ult = latest.cum_paid * cum_f[latest.dev.astype(int) - 1]
        mu = up.ay.map(ult).to_numpy() * beta[up.dev.astype(int) - 1]
        out.append(pd.Series(mu, index=up.index))
    return pd.concat(out)


def ultimates(pred: pd.DataFrame) -> pd.DataFrame:
    """Predicted and actual total ultimate loss per company: all accident years at
    development year 10, as in Kuo (2019) and Meyers (2015)."""
    filled = pred.assign(paid=np.where(pred.upper, pred.incr_paid, pred["pred"]))
    out = filled.groupby(KEY).agg(
        predicted_ultimate=("paid", "sum"), actual_ultimate=("incr_paid", "sum")
    )
    return out.reset_index()


def reserves(pred: pd.DataFrame) -> pd.DataFrame:
    low = pred[~pred.upper]
    r = low.groupby(KEY).agg(predicted_reserve=("pred", "sum"), actual_reserve=("incr_paid", "sum"))
    r["premium"] = pred[pred.dev == 1].groupby(KEY).premium.sum()
    return r.reset_index()


# --- Neural ODP (CANN) ---------------------------------------------------------------------------


class NeuralODP(nn.Module):
    def __init__(self, n_comp: int, n_lob: int, emb: int, hidden: list[int]):
        super().__init__()
        self.comp = nn.Embedding(n_comp, emb)
        self.lob = nn.Embedding(n_lob, emb)
        self.dev = nn.Embedding(10, emb)
        sizes = [3 * emb + 1, *hidden]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.out = nn.Linear(hidden[-1], 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, c, lob, dev, ay, offset):
        z = torch.cat([self.comp(c), self.lob(lob), self.dev(dev), ay[:, None]], 1)
        for layer in self.hidden:
            z = torch.tanh(layer(z))
        return torch.exp(self.out(z)).flatten() * offset


def _odp_frame(tri: pd.DataFrame) -> pd.DataFrame:
    """Each cell with its chain-ladder mean `mu_cl`: the fitted value in the upper triangle,
    the projection below it."""
    d, _ = chain_ladder(tri)
    d.loc[d.upper, "mu_cl"] = chain_ladder_fitted_upper(d)
    d.loc[~d.upper, "mu_cl"] = d.loc[~d.upper, "pred"]
    return d


def neural_odp(tri: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict]:
    full = _odp_frame(tri)
    # as known at the end of 1996: offsets and training cells from calendar years <= 1996
    early = _odp_frame(tri.assign(upper=tri.upper & (tri.calendar < LAST_YEAR)))
    # scale each company by its premium so one network can serve every company (ODP dispersion)
    scale = full[full.dev == 1].groupby(KEY).premium.sum().rename("scale")
    full, early = full.join(scale, on=KEY), early.join(scale, on=KEY)
    codes = {c: pd.Categorical(full[c]).codes for c in ("company", "lob")}
    sizes = (int(codes["company"].max()) + 1, int(codes["lob"].max()) + 1)

    def tensors(d, m):
        return (
            torch.as_tensor(codes["company"][m], dtype=torch.long),
            torch.as_tensor(codes["lob"][m], dtype=torch.long),
            torch.as_tensor(d.dev.to_numpy()[m] - 1, dtype=torch.long),
            torch.as_tensor(((d.ay.to_numpy()[m] - 1992.5) / 3).astype(np.float32)),
            torch.as_tensor((d.mu_cl.clip(lower=1e-6) / d.scale).to_numpy()[m].astype(np.float32)),
            torch.as_tensor((d.incr_paid.clip(lower=0) / d.scale).to_numpy()[m].astype(np.float32)),
        )

    def fit(train, val, epochs):
        torch.manual_seed(params.get("seed", 1))
        net = NeuralODP(*sizes, params.get("embedding_dim", 2), params.get("hidden", [16, 8]))
        dev = device()
        net.to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=params.get("lr", 0.001))
        tr = tuple(t.to(dev) for t in train)
        va = tuple(t.to(dev) for t in val) if val is not None else None
        loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="mean")
        hist = {"loss": [], "val_loss": []}
        for _ in range(epochs):
            net.train()
            opt.zero_grad()
            loss = loss_fn(net(*tr[:5]), tr[5])
            loss.backward()
            opt.step()
            hist["loss"].append(loss.item())
            if va is not None:
                net.eval()
                with torch.no_grad():
                    hist["val_loss"].append(loss_fn(net(*va[:5]), va[5]).item())
        return net.cpu(), hist

    # 1. choose the number of epochs out of time: train <= 1996, validate on the 1997 diagonal
    # (accident year 1997 has no data at the end of 1996, so no offset: left out)
    is_val = ((early.calendar == LAST_YEAR) & early.mu_cl.notna()).to_numpy()
    _, hist = fit(tensors(early, early.upper.to_numpy()), tensors(early, is_val), params["epochs"])
    # validation loss before any training: the chain-ladder offset alone
    base = nn.PoissonNLLLoss(log_input=False, reduction="mean")(
        tensors(early, is_val)[4], tensors(early, is_val)[5]
    ).item()
    best = int(np.argmin(hist["val_loss"]))
    epochs = best + 1 if hist["val_loss"][best] < base else 0
    hist.update(best_epoch=best, epochs_refit=epochs, val_loss_offset_only=base)
    # 2. refit on the whole upper triangle for that many epochs and project the lower triangle
    net, refit = fit(tensors(full, full.upper.to_numpy()), None, epochs)
    hist["refit_loss"] = refit["loss"]
    low = (~full.upper).to_numpy()
    net.eval()
    with torch.no_grad():
        p = net(*tensors(full, low)[:5]).numpy() * full.scale.to_numpy()[low]
    full["pred"] = np.nan
    full.loc[low, "pred"] = p
    return full.drop(columns=["mu_cl", "scale"]), hist
