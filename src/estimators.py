"""Rigobon-Sack pairwise heteroskedasticity estimators."""
import numpy as np


def second_moment(x):
    x = np.asarray(x, float)
    return float(x @ x / len(x))


def cross_moment(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(x @ y / len(x))


def iv_no_intercept(y, x, z):
    """2SLS coefficient and homoskedastic SE for one regressor, k instruments."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[:, None]
    ztz_inv = np.linalg.pinv(z.T @ z)
    xpzx = float(x.T @ z @ ztz_inv @ z.T @ x)
    if abs(xpzx) < 1e-14:
        return np.nan, np.nan, np.nan
    beta = float((x.T @ z @ ztz_inv @ z.T @ y) / xpzx)
    resid = y - beta * x
    dof = max(len(y) - 1, 1)
    sigma2 = float(resid @ resid / dof)
    se = float(np.sqrt(sigma2 / xpzx))
    t = beta / se if se > 0 else np.nan
    return beta, se, t


def estimate_pair(x1_h, x1_l, xj_h, xj_l):
    n = len(x1_h)
    if not (n == len(x1_l) == len(xj_h) == len(xj_l)):
        raise ValueError("H and L vectors must be equal length")
    x1 = np.r_[x1_h, x1_l]
    xj = np.r_[xj_h, xj_l]
    sign = np.r_[np.ones(n), -np.ones(n)]
    w1, w2 = sign * x1, sign * xj
    b1, se1, t1 = iv_no_intercept(xj, x1, w1)
    b2, se2, t2 = iv_no_intercept(xj, x1, w2)
    b3, se3, t3 = iv_no_intercept(xj, x1, np.c_[w1, w2])
    return {"w1_beta": b1, "w1_se": se1, "w1_t": t1,
            "w2_beta": b2, "w2_se": se2, "w2_t": t2,
            "w3_beta": b3, "w3_se": se3, "w3_t": t3,
            "delta_var_x1": second_moment(x1_h)-second_moment(x1_l),
            "var_h_xj": second_moment(xj_h), "var_l_xj": second_moment(xj_l)}
