"""Time-varying hedge ratio via Kalman filter.

State:       x_t = (alpha_t, beta_t)', random walk: x_t = x_{t-1} + w_t
Observation: logp_t = [1, logq_t] x_t + v_t

delta controls state noise (how fast beta may drift); r is observation
noise variance. Returns filtered states and one-step-ahead prediction
errors (the natural online "spread" that avoids look-ahead).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class KalmanOut:
    alpha: np.ndarray
    beta: np.ndarray
    innov: np.ndarray        # one-step prediction error e_t
    innov_std: np.ndarray    # sqrt of predicted innovation variance


def kalman_hedge(logp: np.ndarray, logq: np.ndarray,
                 delta: float = 1e-5, r: float = 1e-3) -> KalmanOut:
    n = len(logp)
    x = np.zeros(2)                        # (alpha, beta)
    P = np.eye(2) * 1.0                    # diffuse-ish prior
    Q = np.eye(2) * (delta / (1.0 - delta))
    alpha = np.empty(n); beta = np.empty(n)
    innov = np.empty(n); innov_std = np.empty(n)

    for t in range(n):
        H = np.array([1.0, logq[t]])
        # predict
        P = P + Q
        yhat = H @ x
        S = H @ P @ H + r
        e = logp[t] - yhat
        # update
        K = P @ H / S
        x = x + K * e
        P = P - np.outer(K, H @ P)
        alpha[t], beta[t] = x
        innov[t] = e
        innov_std[t] = np.sqrt(S)

    return KalmanOut(alpha=alpha, beta=beta, innov=innov, innov_std=innov_std)
