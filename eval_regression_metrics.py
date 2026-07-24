# src/eval_regression_metrics.py
import numpy as np


def _to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _softmax(logits, axis=-1):
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + 1e-12)


def logits3_to_continuous(logits_3, dataset=None, temperature=1.0):
    """
    3-class logits -> continuous sentiment
    - MOSI/MOSEI: fixed map to [-3, 3]
    - else: returns [-1, 1]
    temperature: softmax temperature (T). T<1 sharper, T>1 softer.
    """
    logits_3 = _to_np(logits_3).astype(np.float32)
    T = float(max(1e-6, temperature))
    probs = _softmax(logits_3 / T, axis=1)   # (N,3)

    # direction score in [-1,1]
    s = probs[:, 2] - probs[:, 0]

    if dataset in ["MOSI", "MOSEI"]:
        return (3.0 * s).astype(np.float32)
    return s.astype(np.float32)


def mae(y_pred, y_true):
    y_pred = _to_np(y_pred).reshape(-1).astype(np.float32)
    y_true = _to_np(y_true).reshape(-1).astype(np.float32)
    return float(np.mean(np.abs(y_pred - y_true)))


def pearson_corr(y_pred, y_true):
    y_pred = _to_np(y_pred).reshape(-1).astype(np.float32)
    y_true = _to_np(y_true).reshape(-1).astype(np.float32)

    vx = y_pred - np.mean(y_pred)
    vy = y_true - np.mean(y_true)
    denom = (np.sqrt(np.sum(vx * vx)) * np.sqrt(np.sum(vy * vy))) + 1e-12
    return float(np.sum(vx * vy) / denom)


def fit_linear_calibration(y_pred, y_true):
    """
    Fit y_cal = a*y_pred + b on validation set (least squares).
    """
    y_pred = _to_np(y_pred).reshape(-1).astype(np.float32)
    y_true = _to_np(y_true).reshape(-1).astype(np.float32)

    A = np.vstack([y_pred, np.ones_like(y_pred)]).T  # (N,2)
    (a, b), *_ = np.linalg.lstsq(A, y_true, rcond=None)
    return float(a), float(b)


def apply_linear_calibration(y_pred, a, b):
    y_pred = _to_np(y_pred).reshape(-1).astype(np.float32)
    return (a * y_pred + b).astype(np.float32)


def eval_corr_mae_from_logits3(logits_3, y_true_cont, dataset=None, temperature=1.0):
    """
    logits_3: (N,3)
    y_true_cont: (N,)
    returns Corr/MAE (RAW, no calibration)
    """
    y_true_cont = _to_np(y_true_cont).reshape(-1).astype(np.float32)
    y_pred_cont = logits3_to_continuous(logits_3, dataset=dataset, temperature=temperature)

    return {
        "Corr": round(pearson_corr(y_pred_cont, y_true_cont), 4),
        "MAE": round(mae(y_pred_cont, y_true_cont), 4),
    }


def eval_corr_mae_calibrated_from_logits3(
    logits_valid, y_valid,
    logits_test, y_test,
    dataset=None, temperature=1.0
):
    """
    Fit calibration on (valid), apply to (test).
    Returns:
      metrics_raw_test, metrics_cal_test, (a,b)
    """
    y_valid = _to_np(y_valid).reshape(-1).astype(np.float32)
    y_test = _to_np(y_test).reshape(-1).astype(np.float32)

    pred_valid = logits3_to_continuous(logits_valid, dataset=dataset, temperature=temperature)
    pred_test = logits3_to_continuous(logits_test, dataset=dataset, temperature=temperature)

    a, b = fit_linear_calibration(pred_valid, y_valid)
    pred_test_cal = apply_linear_calibration(pred_test, a, b)

    raw = {
        "Corr": round(pearson_corr(pred_test, y_test), 4),
        "MAE": round(mae(pred_test, y_test), 4),
    }
    cal = {
        "Corr": round(pearson_corr(pred_test_cal, y_test), 4),
        "MAE": round(mae(pred_test_cal, y_test), 4),
    }
    return raw, cal, (a, b)