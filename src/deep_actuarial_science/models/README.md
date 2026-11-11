# `models/`

Model code, one module per domain, independent of Kedro: each fits on data frames and returns
predictions, so a notebook can call it directly.

```mermaid
flowchart TD
    tu["torch_utils.py<br/>seeding, device, Keras init,<br/>Poisson training loop"]
    subgraph freq["frequency.py: freMTPL2"]
        hom[homogeneous] --- glm[GLM / GLM3] --- gbm[GBM]
        fnn[FNN + embeddings] --- cann[CANN on GLM3] --- lgn[LocalGLMnet]
    end
    subgraph mort["mortality.py: Swiss HMD"]
        lc[Lee-Carter] --- nets[CNN, non-linear CNN,<br/>LSTM, Transformer]
    end
    subgraph res["reserving.py: CAS Schedule P"]
        cl[chain-ladder] --> odp[neural ODP<br/>CL offset + network]
    end
    tu --> freq
    tu --> mort
    tu --> res
    init["__init__.py"]
```

| File | Role |
| --- | --- |
| `__init__.py` | package marker |
| `torch_utils.py` | `seed_everything`, `device` (MPS/CUDA/CPU), Keras initialisers, `train_poisson` with early stopping |
| `frequency.py` | homogeneous, GLM, GBM (LightGBM), FNN, CANN, LocalGLMnet; multi-seed networks average their runs |
| `mortality.py` | Lee-Carter (SVD + drift) and four networks on 10-year windows of log m, forecast recursively |
| `reserving.py` | chain-ladder and neural ODP; epochs chosen out of time on the 1997 diagonal |
