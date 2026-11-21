"""Cyber-attack injection for the connected corridor.

Attacks act on two interfaces:

* **Physical / actuation** -- returned as a :class:`StepOverride` applied inside
  the CTM step (green-time starvation, plaza receiving collapse).
* **Perception / V2X** -- applied as corruption of the sensor measurement dict
  (RSU BSM speed spoofing / false data injection, spoofed detour advisory).

Each scenario also declares the ground-truth *compromised nodes* so quarantine
decisions can be scored for correctness.  ``compound`` is a held-out,
zero-shot combination of A+B+C that is never used to fit any estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sumo_env.corridor_sim import StepOverride


def rsu_node(seg: int) -> str:
    return f"RSU{seg}"


def det_node(seg: int) -> str:
    return f"DET{seg}"


def sig_node(inter: int) -> str:
    return f"SIG{inter}"


PLAZA_NODE = "PLAZA"


@dataclass
class AttackInjector:
    cfg: dict
    scenario: str   # 'normal' | 'A' | 'B' | 'C' | 'compound'

    def __post_init__(self) -> None:
        a = self.cfg["attacks"]
        self.onset = float(a["onset_s"])
        self.duration = float(a["duration_s"])
        self.A = a["scenario_A_rsu_spoof"]
        self.B = a["scenario_B_green_starvation"]
        self.C = a["scenario_C_plaza_surge"]

    def active(self, t: float) -> bool:
        return self.onset <= t < self.onset + self.duration

    def _has(self, tag: str) -> bool:
        return self.scenario == tag or (self.scenario == "compound" and tag in ("A", "B", "C"))

    # -- physical actuation override --------------------------------------- #
    def physical_override(self, t: float) -> StepOverride:
        ov = StepOverride()
        if not self.active(t):
            return ov
        if self._has("B"):
            ov.green_split = {int(self.B["target_intersection"]): float(self.B["green_split_attacked"])}
        if self._has("C"):
            ov.plaza_recv_scale = max(0.05, 1.0 - float(self.C["surge_divert_frac"]))
            ov.plaza_divert_frac = min(0.9, self.cfg["dcfc_plaza"]["base_divert_frac"]
                                       + float(self.C["surge_divert_frac"]))
        return ov

    # -- perception corruption --------------------------------------------- #
    def corrupt(self, meas: Dict[str, np.ndarray], t: float) -> Dict[str, np.ndarray]:
        out = {k: v.copy() for k, v in meas.items()}
        if not self.active(t):
            return out
        if self._has("A"):
            seg = int(self.A["target_segment"])
            out["rsu_speed"][seg] = float(self.A["spoof_speed_kmh"])   # falsely free-flowing
        if self._has("C"):
            seg = int(self.C["target_segment"])
            gain = float(self.C["spoofed_advisory_gain"])
            out["det_rho"][seg] = max(0.0, out["det_rho"][seg] * (1.0 - gain))  # under-reported
        return out

    # -- ground-truth compromised nodes ------------------------------------ #
    def compromised_nodes(self, t: float) -> List[str]:
        if not self.active(t):
            return []
        nodes: List[str] = []
        if self._has("A"):
            nodes.append(rsu_node(int(self.A["target_segment"])))
        if self._has("B"):
            nodes.append(sig_node(int(self.B["target_intersection"])))
        if self._has("C"):
            nodes.append(PLAZA_NODE)
            nodes.append(det_node(int(self.C["target_segment"])))
        return nodes

    def compromised_nodes_any(self) -> List[str]:
        """Union over the whole attack window (for scoring / quarantine targets)."""
        return self.compromised_nodes(self.onset)


SCENARIOS = ["normal", "A", "B", "C", "compound"]
SINGLE_ATTACKS = ["A", "B", "C"]
