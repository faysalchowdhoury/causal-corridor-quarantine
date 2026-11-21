"""State reconstructors compared in the benchmark.

* :class:`ExtendedKalmanReconstructor` -- physically grounded EKF whose process
  model is the Godunov/LWR CTM (analytic tridiagonal Jacobian).  Posterior is
  projected onto ``[0, rho_jam]`` so it respects bounds, but it *trusts every
  measurement*, so false-data injection biases it.
* :class:`UnconstrainedNeuralReconstructor` -- PyTorch MLP trained on nominal
  data.  Fast, but output is **not** constrained, so on unseen compound attacks
  it produces ``rho > rho_jam`` / ``rho < 0`` artifacts and large errors.
* :class:`ConservationConstrainedReconstructor` (**ours**) -- predicts with the
  conservation (CTM) model, quarantines compromised sensors, fuses only trusted
  detector/RSU evidence, and projects onto the feasible set, so it stays mass-
  consistent across quarantined sensor gaps.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # pragma: no cover - torch is a hard dependency but degrade gracefully
    _TORCH = False

from sumo_env.corridor_sim import FundamentalDiagram
from cyber_attacks.attack_injector import det_node, rsu_node


# --------------------------------------------------------------------------- #
# Nominal CTM process model (used by the EKF and the constrained reconstructor)
# --------------------------------------------------------------------------- #
class CTMProcessModel:
    """Nominal (attack-agnostic) corridor dynamics with an analytic Jacobian."""

    def __init__(self, cfg: dict):
        c = cfg["corridor"]
        self.n_seg = int(c["n_segments"])
        self.cps = int(c["cells_per_segment"])
        self.N = self.n_seg * self.cps
        self.dx = float(c["segment_length_m"]) / 1000.0 / self.cps
        self.dt = float(c["dt_s"]) / 3600.0
        self.dt_s = float(c["dt_s"])
        fd = cfg["fundamental_diagram"]
        self.fd = FundamentalDiagram(fd["v_free_kmh"], fd["rho_jam_vehkm"], fd["q_max_vehh"])
        sg = cfg["signals"]
        self.cycle = float(sg["cycle_s"])
        self.green_split = float(sg["green_split"])
        self.offset = list(sg["offset_s"])
        self.inflow = float(cfg["demand"]["inflow_vehh"])
        self.warmup_s = float(cfg["demand"]["warmup_s"])

    def _green(self, t: float, seg: int) -> float:
        phase = (t + self.offset[seg]) % self.cycle
        return 1.0 if phase < self.green_split * self.cycle else 0.0

    def _demand(self, t: float) -> float:
        return self.inflow * min(1.0, t / max(self.warmup_s, 1e-6))

    def predict(self, rho: np.ndarray, t: float) -> np.ndarray:
        rho = np.asarray(rho, float)
        S = self.fd.sending(rho)
        R = self.fd.receiving(rho)
        f = np.minimum(S[:-1], R[1:])
        for s in range(self.n_seg):
            j = s * self.cps + self.cps - 1
            if j < self.N - 1:
                f[j] *= self._green(t, s)
        f_up = min(self._demand(t), R[0])
        f_down = S[-1] * self._green(t, self.n_seg - 1)
        drho = np.empty(self.N)
        drho[0] = f_up - f[0]
        drho[1:-1] = f[:-1] - f[1:]
        drho[-1] = f[-1] - f_down
        return np.clip(rho + (self.dt / self.dx) * drho, 0.0, self.fd.rho_jam)

    def jacobian(self, rho: np.ndarray, t: float) -> np.ndarray:
        """Analytic tridiagonal F = d(rho_next)/d(rho) for the triangular FD."""
        rho = np.asarray(rho, float)
        fd = self.fd
        Sp = np.where(rho < fd.rho_crit, fd.v_free, 0.0)       # dS/drho
        Rp = np.where(rho > fd.rho_crit, -fd.w, 0.0)           # dR/drho
        S, R = fd.sending(rho), fd.receiving(rho)
        # boundary flux derivatives df_k/drho_k and df_k/drho_{k+1}
        dfk_dk = np.zeros(self.N - 1)
        dfk_dk1 = np.zeros(self.N - 1)
        send_binds = S[:-1] <= R[1:]
        dfk_dk[send_binds] = Sp[:-1][send_binds]
        dfk_dk1[~send_binds] = Rp[1:][~send_binds]
        for s in range(self.n_seg):
            j = s * self.cps + self.cps - 1
            if j < self.N - 1:
                g = self._green(t, s)
                dfk_dk[j] *= g
                dfk_dk1[j] *= g
        lam = self.dt / self.dx
        F = np.eye(self.N)
        for i in range(self.N):
            if i < self.N - 1:                      # -df_i/drho terms
                F[i, i] -= lam * dfk_dk[i]
                F[i, i + 1] -= lam * dfk_dk1[i]
            if i > 0:                               # +df_{i-1}/drho terms
                F[i, i - 1] += lam * dfk_dk[i - 1]
                F[i, i] += lam * dfk_dk1[i - 1]
        return F


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class Reconstructor:
    name = "base"

    def reset(self, rho0: np.ndarray) -> None:
        self.rho = np.asarray(rho0, float).copy()

    def reconstruct(self, meas: Dict[str, np.ndarray], t: float,
                    quarantine: Optional[Sequence[str]] = None) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# EKF
# --------------------------------------------------------------------------- #
class ExtendedKalmanReconstructor(Reconstructor):
    name = "EKF (LWR Godunov)"

    def __init__(self, cfg: dict):
        self.pm = CTMProcessModel(cfg)
        self.N = self.pm.N
        self.n_seg = self.pm.n_seg
        self.cps = self.pm.cps
        # measurements: segment-mean detector density (4) + RSU-derived density (4)
        self.H = np.zeros((2 * self.n_seg, self.N))
        for s in range(self.n_seg):
            lo = s * self.cps
            self.H[s, lo:lo + self.cps] = 1.0 / self.cps               # detector seg-mean
            self.H[self.n_seg + s, lo:lo + self.cps] = 1.0 / self.cps  # RSU seg-mean
        mv = float(cfg["estimators"]["ekf_meas_var"])
        self.Q = float(cfg["estimators"]["ekf_process_var"]) * np.eye(self.N)
        self.R = np.diag(np.concatenate([np.full(self.n_seg, mv),
                                         np.full(self.n_seg, 3.0 * mv)]))  # RSU inversion noisier
        self.fd = self.pm.fd

    def _rho_from_speed(self, v: np.ndarray) -> np.ndarray:
        return np.clip(self.fd.w * self.fd.rho_jam / (np.asarray(v, float) + self.fd.w),
                       0.0, self.fd.rho_jam)

    def reset(self, rho0: np.ndarray) -> None:
        self.rho = np.asarray(rho0, float).copy()
        self.P = 5.0 * np.eye(self.N)

    def reconstruct(self, meas, t, quarantine=None):
        F = self.pm.jacobian(self.rho, t)
        x = self.pm.predict(self.rho, t)
        self.P = F @ self.P @ F.T + self.Q
        z = np.concatenate([np.asarray(meas["det_rho"], float),
                            self._rho_from_speed(meas["rsu_speed"])])  # trusts every sensor
        y = z - self.H @ x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        x = x + K @ y
        self.P = (np.eye(self.N) - K @ self.H) @ self.P
        self.rho = np.clip(x, 0.0, self.pm.fd.rho_jam)   # physically grounded projection
        return self.rho.copy()


# --------------------------------------------------------------------------- #
# Unconstrained neural baseline
# --------------------------------------------------------------------------- #
def measurement_vector(meas: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([meas["det_rho"], meas["det_flow"], meas["rsu_speed"]]).astype(float)


class UnconstrainedNeuralReconstructor(Reconstructor):
    name = "Unconstrained Neural"

    def __init__(self, cfg: dict):
        if not _TORCH:
            raise RuntimeError("PyTorch is required for the neural baseline.")
        self.cfg = cfg
        self.N = int(cfg["corridor"]["n_segments"]) * int(cfg["corridor"]["cells_per_segment"])
        self.in_dim = 3 * int(cfg["corridor"]["n_segments"])
        h = int(cfg["estimators"]["neural_hidden"])
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, self.N),           # NO output activation -> unconstrained
        )
        self._mu = np.zeros(self.in_dim)
        self._sd = np.ones(self.in_dim)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = np.asarray(X, float)
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self._mu) / self._sd
        xt = torch.tensor(Xn, dtype=torch.float32)
        yt = torch.tensor(np.asarray(Y, float), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg["estimators"]["neural_lr"]))
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(int(self.cfg["estimators"]["neural_epochs"])):
            opt.zero_grad()
            loss = loss_fn(self.net(xt), yt)
            loss.backward()
            opt.step()
        self.net.eval()

    def reconstruct(self, meas, t, quarantine=None):
        x = (measurement_vector(meas) - self._mu) / self._sd
        with torch.no_grad():
            out = self.net(torch.tensor(x, dtype=torch.float32)).numpy()
        self.rho = out                                  # deliberately NOT clipped
        return self.rho.copy()


# --------------------------------------------------------------------------- #
# Conservation-constrained reconstructor (ours)
# --------------------------------------------------------------------------- #
class ConservationConstrainedReconstructor(Reconstructor):
    name = "Causal Quarantine + Conservation (ours)"

    def __init__(self, cfg: dict):
        self.pm = CTMProcessModel(cfg)
        self.N = self.pm.N
        self.n_seg = self.pm.n_seg
        self.cps = self.pm.cps
        self.gain = float(cfg["estimators"]["constrained_meas_gain"])
        self.fd = self.pm.fd

    def _rho_from_speed(self, v: float) -> Optional[float]:
        """Invert the triangular FD on the congested branch (speed is only
        informative when the segment is actually slow)."""
        if v >= self.fd.v_free - 2.0:
            return None
        return float(np.clip(self.fd.w * self.fd.rho_jam / (v + self.fd.w),
                             0.0, self.fd.rho_jam))

    def reconstruct(self, meas, t, quarantine=None):
        q = set(quarantine or [])
        x = self.pm.predict(self.rho, t)                # conservation prediction (fills gaps)
        det_rho = np.asarray(meas["det_rho"], float)
        rsu_speed = np.asarray(meas["rsu_speed"], float)
        for s in range(self.n_seg):
            lo = s * self.cps
            seg_mean = x[lo:lo + self.cps].mean()
            trust_det = det_node(s) not in q
            trust_rsu = rsu_node(s) not in q
            if trust_det:                               # nudge segment toward detector seg-mean
                x[lo:lo + self.cps] += self.gain * (det_rho[s] - seg_mean)
                seg_mean = x[lo:lo + self.cps].mean()
            if trust_rsu:
                rho_v = self._rho_from_speed(rsu_speed[s])
                if rho_v is not None:                   # congested evidence corroborates queue
                    x[lo:lo + self.cps] += 0.4 * (rho_v - seg_mean)
        self.rho = np.clip(x, 0.0, self.fd.rho_jam)     # feasible-set projection
        return self.rho.copy()
