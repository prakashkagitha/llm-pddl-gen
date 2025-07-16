"""
Dual-BFWS-FFparser solver wrapper.

• offline=True  (default)  → run `planutils run dual-bfws-ffparser`
  with a hard time-limit; success = non-empty plan file.

• online=True              → fallback to planning.domains REST API
  (behaviour unchanged from the original implementation).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

from .solver_interface import SolverInterface


class DualBFWSSolver(SolverInterface):
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        solver: str = "dual-bfws-ffparser",
        *,
        online: bool = False,
        workdir: Optional[str | Path] = None,
        plan_filename: str = "plan",
        time_limit: int = 5,          # seconds (adjust as needed)
    ):
        self.solver = solver
        self.online = online
        self.workdir = Path(workdir) if workdir else Path.cwd()
        self.plan_path = self.workdir / plan_filename
        self.time_limit = time_limit

        if self.online:
            self.base_url = "https://solver.planning.domains:5001"

    # ------------------------------------------------------------------ #
    #  public API required by SolverInterface                            #
    # ------------------------------------------------------------------ #
    def solve(self, domain_file: str, problem_file: str):
        plan, _ = self.solve_with_error(domain_file, problem_file)
        return plan

    def solve_with_error(
        self, domain_file: str, problem_file: str
    ) -> Tuple[Optional[str], str]:
        if self.online:
            return self._solve_online(domain_file, problem_file)
        return self._solve_offline(domain_file, problem_file)

    # ------------------------------------------------------------------ #
    #  OFFLINE path: planutils                                           #
    # ------------------------------------------------------------------ #
    def _solve_offline(
        self, domain_file: str, problem_file: str
    ) -> Tuple[Optional[str], str]:
        self.plan_path.unlink(missing_ok=True)

        cmd = ["planutils", "run", self.solver, domain_file, problem_file]
        proc = subprocess.Popen(
            cmd,
            cwd=self.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,      # own process group
        )

        try:
            stdout, stderr = proc.communicate(timeout=self.time_limit)
        except subprocess.TimeoutExpired:
            # ---------- hard timeout ----------
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()

        # ---------- success? ----------
        if self.plan_path.exists():
            plan_txt = self.plan_path.read_text(encoding="utf-8").strip()
            self.plan_path.unlink(missing_ok=True)
            if plan_txt:                       # non-empty → success
                return plan_txt, ""
            # empty file counts as failure → fall through

        err_msg = ((stderr or "") + "\n" + (stdout or "")).strip() or "plan not found"
        return None, err_msg

    # ------------------------------------------------------------------ #
    #  ONLINE path: planning.domains REST API                            #
    # ------------------------------------------------------------------ #
    def _solve_online(
        self, domain_file: str, problem_file: str
    ) -> Tuple[Optional[str], str]:
        with open(domain_file) as f:
            dom_txt = f.read()
        with open(problem_file) as f:
            prob_txt = f.read()

        req = {"domain": dom_txt, "problem": prob_txt}
        try:
            job = requests.post(
                f"{self.base_url}/package/{self.solver}/solve", json=req, timeout=10
            ).json()
        except requests.RequestException as exc:
            return None, f"submit failed: {exc}"

        result_url = self.base_url + job["result"]
        while True:
            try:
                res = requests.post(result_url, timeout=10).json()
            except requests.RequestException as exc:
                return None, f"poll failed: {exc}"
            if res.get("status") != "PENDING":
                break
            time.sleep(0.5)

        if "Error" in res:
            return None, "timeout"

        result = res["result"]
        # planning.domains’ dual-bfws response quirks
        if self.solver == "dual-bfws-ffparser" and result.get("output") == {"plan": ""}:
            msg = result["stderr"] or result["stdout"]
            return None, msg or "plan not found"

        # success
        out = result["output"]
        plan = out["plan"] if isinstance(out, dict) and "plan" in out else out
        return plan, ""