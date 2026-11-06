"""Claim-frequency models on freMTPL2: homogeneous, GLM, GBM, FNN, CANN, LocalGLMnet.

Every model exposes `fit(learn, params, base=None) -> self` and
`predict(df) -> expected claim counts` (already multiplied by exposure).
The preprocessing and architectures follow Wüthrich et al., "AI Tools for
Actuaries" (2025), so results reconcile to the published notebooks.

Add a model: subclass `FrequencyModel`, register it in `MODELS`, add a
parameter block under `frequency.models` and (optionally) a benchmark.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from statsmodels.formula.api import glm as smf_glm
from torch import nn

from deep_actuarial_science.models.torch_utils import (
    n_params,
    seed_everything,
    tensor,
    train_poisson,
)

DRIV_AGE_LEVELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]


# --- Feature engineering -----------------------------------------------------------------


def add_glm_features(df: pd.DataFrame) -> pd.DataFrame:
    """GLM covariates of Wüthrich & Merz (2023), Section 5.2.4."""
    ref_first = ["31-40", *[lev for lev in DRIV_AGE_LEVELS if lev != "31-40"]]
    return df.assign(
        AreaGLM=df.Area.cat.codes + 1,
        VehPowerGLM=pd.Categorical(df.VehPower.clip(0, 9)),
        VehAgeGLM=pd.cut(df.VehAge, bins=[-1, 5, 12, 101], labels=["0-5", "6-12", "12+"]),
        DrivAgeGLM=pd.Categorical(
            pd.cut(df.DrivAge, bins=[17, 20, 25, 30, 40, 50, 70, 101], labels=DRIV_AGE_LEVELS),
            categories=ref_first,
        ),
        BonusMalusGLM=df.BonusMalus.clip(0, 150),
        DensityGLM=np.log(df.Density),
    )


def _log_round(x):
    return np.log(x).round(2)


def _is_diesel(x):
    return (x == "Diesel").astype(np.float32)


class _Clip:
    def __init__(self, upper):
        self.upper = upper

    def __call__(self, x):
        return np.clip(x, a_min=0, a_max=self.upper)


def _clip_scale(upper):
    return Pipeline([("clip", FunctionTransformer(_Clip(upper))), ("scale", StandardScaler())])


NUMERIC = ["VehAge", "DrivAge", "BonusMalus", "VehPower"]
CLIP_UPPER = [20, 90, 150, 15]
# Output column order of nn_preprocessor(embed=False)
DESIGN_COLUMNS = [
    *NUMERIC,
    "Area",
    "Density",
    "VehGas",
    "VehBrandEmb1",
    "VehBrandEmb2",
    "RegionEmb1",
    "RegionEmb2",
    "RandN",
]


def nn_preprocessor(embed: bool) -> ColumnTransformer:
    """Scaled continuous covariates; VehBrand/Region as integer codes (embed) or PCA'd
    pre-trained embeddings (LocalGLMnet, which needs a numeric design matrix)."""
    common = [
        ("scale", _clip_scale(CLIP_UPPER), NUMERIC),
        ("area", Pipeline([("enc", OrdinalEncoder()), ("scale", StandardScaler())]), ["Area"]),
        (
            "density",
            Pipeline([("log", FunctionTransformer(_log_round)), ("scale", StandardScaler())]),
            ["Density"],
        ),
        ("veh_gas", FunctionTransformer(_is_diesel), ["VehGas"]),
    ]
    if embed:
        cats = [("cats", OrdinalEncoder(), ["VehBrand", "Region"])]
    else:
        cats = [
            (
                f"pca_{n}",
                Pipeline([("pca", PCA(n_components=2)), ("scale", StandardScaler())]),
                [f"{n}Emb1", f"{n}Emb2"],
            )
            for n in ("VehBrand", "Region")
        ] + [("rand", "passthrough", ["RandN"])]
    return ColumnTransformer(common + cats, verbose_feature_names_out=False)


def split_train_val(learn: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        learn, test_size=params.get("val_size", 0.1), random_state=params.get("split_seed", 125548)
    )


# --- Models --------------------------------------------------------------------------------


class FrequencyModel:
    name = "base"
    history: dict | None = None
    fit_seconds: float = 0.0

    def fit(self, learn: pd.DataFrame, params: dict, base: FrequencyModel | None = None):
        start = time.perf_counter()
        self._fit(learn, params, base)
        self.fit_seconds = time.perf_counter() - start
        return self

    def _fit(self, learn, params, base):
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def n_params(self) -> int:
        return 0


class Homogeneous(FrequencyModel):
    name = "homogeneous"

    def _fit(self, learn, params, base):
        self.lam = learn.ClaimNb.sum() / learn.Exposure.sum()

    def predict(self, df):
        return self.lam * df.Exposure.to_numpy()

    def n_params(self):
        return 1


class GLM(FrequencyModel):
    """Poisson GLM with log link and log-exposure offset (statsmodels formula)."""

    name = "glm"

    def _fit(self, learn, params, base):
        d = add_glm_features(learn)
        self.formula = params["formula"]
        self.result = smf_glm(
            self.formula, data=d, offset=np.log(d.Exposure), family=sm.families.Poisson()
        ).fit()

    def predict(self, df):
        d = add_glm_features(df)
        return self.result.predict(d).to_numpy() * df.Exposure.to_numpy()

    def n_params(self):
        return len(self.result.params)


class GBM(FrequencyModel):
    """LightGBM with Poisson objective and log-exposure as init score (offset)."""

    name = "gbm"
    features = [
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Density",
        "Region",
        "Area",
    ]

    def _fit(self, learn, params, base):
        train, val = split_train_val(learn, params)
        p = {
            "objective": "poisson",
            "verbose": -1,
            "seed": params.get("seed", 1),
            "deterministic": True,
            "num_threads": params.get("num_threads", 0),
            **params.get("lgb", {}),
        }
        # LightGBM skips boost-from-average when an init_score is given: start the trees at
        # the portfolio frequency (log mu_hom) plus the log-exposure offset.
        self.log_mu0 = float(np.log(train.ClaimNb.sum() / train.Exposure.sum()))
        dtr = lgb.Dataset(
            train[self.features], train.ClaimNb, init_score=np.log(train.Exposure) + self.log_mu0
        )
        dva = lgb.Dataset(
            val[self.features], val.ClaimNb, init_score=np.log(val.Exposure) + self.log_mu0
        )
        evals: dict = {}
        self.booster = lgb.train(
            p,
            dtr,
            num_boost_round=params.get("num_boost_round", 2000),
            valid_sets=[dtr, dva],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(params.get("early_stopping", 50), verbose=False),
                lgb.record_evaluation(evals),
            ],
        )
        self.history = {
            "loss": evals["train"]["poisson"],
            "val_loss": evals["val"]["poisson"],
            "best_epoch": self.booster.best_iteration - 1,
        }

    def predict(self, df):
        raw = self.booster.predict(
            df[self.features], num_iteration=self.booster.best_iteration, raw_score=True
        )
        return np.exp(raw + self.log_mu0) * df.Exposure.to_numpy()

    def n_params(self):
        return int(self.booster.num_trees())


class EmbeddingFNN(nn.Module):
    """Depth-d tanh FNN with entity embeddings for VehBrand and Region.

    Output = exp(w0 + <w, z(x)>) * offset. With w = 0 at initialisation the
    network starts at exp(w0) * offset: the homogeneous model (FNN) or the
    GLM (CANN, w0 = 0 and offset = GLM expected counts).
    """

    def __init__(
        self, n_cont: int, n_levels: list[int], emb_dims: list[int], hidden: list[int], bias0: float
    ):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(k, b) for k, b in zip(n_levels, emb_dims))
        sizes = [n_cont + sum(emb_dims), *hidden]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.out = nn.Linear(hidden[-1], 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, bias0)

    def features(self, x):
        cats = [e(x[:, -len(self.emb) + i].long()) for i, e in enumerate(self.emb)]
        z = torch.cat([x[:, : -len(self.emb)], *cats], dim=1)
        for layer in self.hidden:
            z = torch.tanh(layer(z))
        return z

    def forward(self, x, offset):
        return torch.exp(self.out(self.features(x))).flatten() * offset


class FNN(FrequencyModel):
    name = "fnn"
    uses_base = False

    def _offset(self, df, base):
        return base.predict(df) if self.uses_base else df.Exposure.to_numpy()

    def _fit(self, learn, params, base):
        seed_everything(params.get("seed", 21456783))
        self.base = base
        self.pre = nn_preprocessor(embed=True).fit(learn)
        train, val = split_train_val(learn, params)
        n_levels = [learn.VehBrand.nunique(), learn.Region.nunique()]
        bias0 = 0.0 if self.uses_base else float(np.log(learn.ClaimNb.sum() / learn.Exposure.sum()))
        torch.manual_seed(params.get("seed", 21456783))
        self.net = EmbeddingFNN(
            len(NUMERIC) + 3,
            n_levels,
            params.get("embedding_dims", [2, 2]),
            params.get("hidden", [20, 15, 10]),
            bias0,
        )
        self.history = train_poisson(
            self.net,
            self._tensors(train),
            self._tensors(val),
            epochs=params["epochs"],
            batch_size=params.get("batch_size", 5000),
            lr=params.get("lr"),
            shuffle=params.get("shuffle", False),
        )
        if params.get("balance_correction", False):
            self._balance(learn)

    def _tensors(self, df):
        return (
            tensor(self.pre.transform(df)),
            tensor(df.ClaimNb),
            tensor(self._offset(df, self.base)),
            tensor(df.Exposure),
        )

    def _balance(self, learn):
        """Restore the balance property by shifting the output bias (Wüthrich 2020)."""
        ratio = learn.ClaimNb.sum() / self.predict(learn).sum()
        with torch.no_grad():
            self.net.out.bias += float(np.log(ratio))

    def predict(self, df):
        self.net.eval()
        X, _, off, _ = self._tensors(df)
        with torch.no_grad():
            return self.net(X, off).numpy().astype(float)

    def n_params(self):
        return n_params(self.net)


class CANN(FNN):
    """Combined Actuarial Neural Network (Wüthrich & Merz 2019): GLM prediction as offset."""

    name = "cann"
    uses_base = True


class LocalGLMnetNet(nn.Module):
    """log mu(x) = beta0 + <beta(x), x>, beta(x) = network output (Richman & Wüthrich 2023)."""

    def __init__(self, n_features: int, hidden: list[int], intercept: float, glm_coefs):
        super().__init__()
        sizes = [n_features, *hidden]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.attention = nn.Linear(hidden[-1], n_features)
        nn.init.zeros_(self.attention.weight)
        self.attention.bias.data = torch.as_tensor(np.asarray(glm_coefs, dtype=np.float32))
        self.intercept = nn.Parameter(torch.tensor(float(intercept)))

    def betas(self, x):
        z = x
        for layer in self.hidden:
            z = torch.tanh(layer(z))
        return self.attention(z)

    def forward(self, x, offset):
        return torch.exp(self.intercept + (self.betas(x) * x).sum(1)) * offset


class LocalGLMnet(FrequencyModel):
    name = "localglmnet"

    def design(self, df) -> pd.DataFrame:
        X = pd.DataFrame(self.pre.transform(df), columns=DESIGN_COLUMNS, index=df.index)
        return X[self.features].astype(np.float32)

    def _design(self, df):
        return self.design(df).to_numpy()

    def _fit(self, learn, params, base):
        self.pre = nn_preprocessor(embed=False).fit(learn)
        self.features = params["features"]
        X = pd.DataFrame(self._design(learn), columns=self.features)
        glm0 = sm.GLM(
            learn.ClaimNb.to_numpy(),
            sm.add_constant(X),
            offset=np.log(learn.Exposure),
            family=sm.families.Poisson(),
        ).fit()
        self.glm_baseline = glm0
        seed_everything(params.get("seed", 21456783))
        torch.manual_seed(params.get("seed", 21456783))
        self.net = LocalGLMnetNet(
            len(self.features),
            params.get("hidden", [20, 15, 10]),
            glm0.params.iloc[0],
            glm0.params.to_numpy()[1:],
        )
        train, val = split_train_val(learn, params)
        self.history = train_poisson(
            self.net,
            self._tensors(train),
            self._tensors(val),
            epochs=params["epochs"],
            batch_size=params.get("batch_size", 5000),
            lr=params.get("lr"),
            shuffle=params.get("shuffle", False),
        )

    def _tensors(self, df):
        return (
            tensor(self._design(df)),
            tensor(df.ClaimNb),
            tensor(df.Exposure),
            tensor(df.Exposure),
        )

    def predict(self, df):
        self.net.eval()
        with torch.no_grad():
            return self.net(tensor(self._design(df)), tensor(df.Exposure)).numpy().astype(float)

    def attentions(self, df) -> pd.DataFrame:
        self.net.eval()
        with torch.no_grad():
            b = self.net.betas(tensor(self._design(df))).numpy()
        return pd.DataFrame(b, columns=self.features, index=df.index)

    def n_params(self):
        return n_params(self.net)


MODELS: dict[str, type[FrequencyModel]] = {
    m.name: m for m in (Homogeneous, GLM, GBM, FNN, CANN, LocalGLMnet)
}


def build(kind: str) -> FrequencyModel:
    return MODELS[kind]()
