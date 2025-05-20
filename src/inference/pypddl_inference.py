# src/inference/pypddl_inference.py
"""
PyPDDLInference
---------------
• Saves python domain / problem scripts plus the resulting PDDL.
• Extra metrics:
      – python_exec_ok   (bool per-problem)
      – parse_ok         (bool per-problem)
  + aggregate percentages written to metrics_pypddl_summary.csv
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Tuple

from py2pddl import parse
from .base_inference import BaseInference


class PyPDDLInference(BaseInference):
    # ------------------------------------------------------------------ #
    #  Small helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prefix() -> str:
        """Common prompt prefix (static because it never changes)."""
        with open("prompts/pypddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    def _descs(self, pid: str) -> Tuple[str, str]:
        """Load natural-language descriptions for a single problem."""
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f:
            dom = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f:
            prob = f.read()
        return dom, prob

    # ------------------------------------------------------------------ #
    #  Override: multi-stage generation                                  #
    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        """
        One candidate per problem.

        • Creates `<out_dir>/<pid>/…` folders up front so we can
          write the Python sources there before `BaseInference`
          evaluates the PDDL syntactic / semantic correctness.
        """
        prompts: List[str] = []
        for pid in pids:
            dom, prob = self._descs(pid)
            prompts.append(
                self._prefix()
                + f"Domain description:\n{dom}\n\n"
                + f"Problem description:\n{prob}\n"
                + "Write the domain and problem files in minimal **PyPDDL**.\n"
                + "Wrap the Python *domain + problem* inside <domain_file>…</domain_file> "
                + "and any *additional* problem-specific helpers inside "
                "<problem_file>…</problem_file>.\n<think>"
            )

        outs = self.llm.generate(prompts, self.sampler)

        results: List[List[Dict]] = []
        # will be consumed later by evaluate()
        self.extra_rows: List[Dict[str, bool]] = []

        for pid, o in zip(pids, outs):
            txt = o.outputs[0].text
            dom_py = self._unwrap(txt, "domain_file")
            prob_py = self._unwrap(txt, "problem_file")

            # -------- write sources inside the experiment out_dir -------- #
            prob_root = os.path.join(self.out_dir, pid)      # e.g. …/p03
            os.makedirs(prob_root, exist_ok=True)

            py_src_path = os.path.join(prob_root, "domain_problem.py")
            prob_py_path = os.path.join(prob_root, "problem.py")
            with open(py_src_path, "w") as f:
                f.write(dom_py + "\n\n" + prob_py)
            with open(prob_py_path, "w") as f:
                f.write(prob_py)

            # -------- run py2pddl --------------------------------------- #
            domain_pddl_path = os.path.join(prob_root, "domain.pddl")
            problem_pddl_path = os.path.join(prob_root, "problem.pddl")

            parse_ok = python_ok = True
            try:
                parse.parse(
                    py_src_path,
                    domain=domain_pddl_path,
                    problem=problem_pddl_path,
                )
            except Exception:
                parse_ok = False

            # simple exec test
            try:
                compile(dom_py + "\n" + prob_py, "<string>", "exec")
            except Exception:
                python_ok = False

            # ---------- bookkeeping ------------------------------------- #
            self.extra_rows.append(
                {"problem": pid, "python_exec_ok": python_ok, "parse_ok": parse_ok}
            )

            # parse.parse adds .pddl to the paths
            domain_pddl_path += ".pddl"
            problem_pddl_path += ".pddl"
            # ---------- produce candidate dict for BaseInference --------- #
            if parse_ok and os.path.exists(domain_pddl_path) and os.path.exists(
                problem_pddl_path
            ):
                with open(domain_pddl_path) as d:
                    df = d.read()
                with open(problem_pddl_path) as p:
                    pf = p.read()
            else:
                df = pf = ""

            results.append([{"df": df, "pf": pf, "raw": txt}])

        return results

    # ------------------------------------------------------------------ #
    #  Override: evaluate to add extra CSVs & percentages                 #
    # ------------------------------------------------------------------ #
    def evaluate(self, pids: List[str], out_dir: str) -> Tuple[float, float]:
        """
        • Stores `out_dir` so batch_generate_candidates can see it.
        • Saves per-problem metrics and a one-line summary CSV.
        """
        self.out_dir = out_dir  # <-- used everywhere else
        self.extra_rows = []    # start fresh per experiment

        # run the standard syntactic / semantic checks
        syn_acc, sem_acc = super().evaluate(pids, out_dir)

        # ---------------- write per-problem CSV ------------------------ #
        per_prob_path = os.path.join(out_dir, "metrics_pypddl.csv")
        if self.extra_rows:
            with open(per_prob_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.extra_rows[0].keys())
                writer.writeheader()
                writer.writerows(self.extra_rows)

        # ---------------- write tiny summary CSV ----------------------- #
        n = len(self.extra_rows) or 1  # avoid divide-by-zero
        py_exec_acc = sum(r["python_exec_ok"] for r in self.extra_rows) / n
        parse_acc = sum(r["parse_ok"] for r in self.extra_rows) / n

        summary_path = os.path.join(out_dir, "metrics_pypddl_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "python_exec_accuracy",
                    "parse_accuracy",
                    "syntactic_accuracy",
                    "semantic_accuracy",
                    "n_problems",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "python_exec_accuracy": py_exec_acc,
                    "parse_accuracy": parse_acc,
                    "syntactic_accuracy": syn_acc,
                    "semantic_accuracy": sem_acc,
                    "n_problems": len(pids),
                }
            )

        return syn_acc, sem_acc

    # ------------------------------------------------------------------ #
    #  Not used (BaseInference calls batch_generate_candidates directly) #
    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "PyPDDLInference overrides batch_generate_candidates, "
            "so get_prompt should never be called."
        )
