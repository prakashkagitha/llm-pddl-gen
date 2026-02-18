"""
Single-stage PDDL generation pipeline.

This pipeline mirrors :class:`PDDLKnowledgeInference` but follows the
newer pipeline style: prompts rely on ``prompts/only_pddl_instruction``
and always surface the original domain/problem descriptions before
requesting final PDDL artefacts.  Artefacts are saved via the shared
``BaseInference`` helpers, so downstream tooling can analyse solver
metrics identically to other pipelines.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

from .base_inference import BaseInference


class PDDLInference(BaseInference):
    """One-pass PDDL generation from natural-language descriptions."""

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pddl_prefix() -> str:
        with open("prompts/only_pddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    def _descs(self, pid: str) -> Tuple[str, str]:
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f:
            dom = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f:
            prob = f.read()
        return dom, prob

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:
        domain_desc, problem_desc = self._descs(pid)
        return (
            self._pddl_prefix()
            + "You are an expert PDDL engineer.\n"
            "Create syntactically correct and mutually consistent domain and "
            "problem files using the information below.\n\n"
            "### Original domain description\n"
            f"{domain_desc}\n\n"
            "### Original problem description\n"
            f"{problem_desc}\n\n"
            "Wrap only the final domain PDDL in <domain_file>…</domain_file> and the "
            "final problem PDDL in <problem_file>…</problem_file>."
        )

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        syn_ok = sem_ok = 0
        plan_not_found = plan_not_valid = 0

        per_problem_rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            s_ok = se_ok = False
            plan_found_for_problem = False
            final_category = ""
            last_solver_error = ""
            last_validator_msg = ""
            last_candidate_idx = -1

            for idx, cand in enumerate(cand_list):
                so, seo, sol_err, val_msg, plan_found, category = self._check_candidate(
                    pid, idx, cand, out_dir, stage_name="stage1_final_pddl"
                )
                cand_dir = os.path.join(out_dir, pid, f"cand_{idx:02}")
                if sol_err:
                    with open(os.path.join(cand_dir, "solver_error.txt"), "w") as f:
                        f.write(sol_err)
                if val_msg:
                    with open(os.path.join(cand_dir, "validator.log"), "w") as f:
                        f.write(val_msg)

                last_candidate_idx = idx
                last_solver_error = sol_err
                last_validator_msg = val_msg

                s_ok |= so
                se_ok |= seo
                plan_found_for_problem |= plan_found

                if category:
                    final_category = category
                if se_ok:
                    break

            per_problem_rows.append(
                {
                    "problem": pid,
                    "used_candidate_index": last_candidate_idx,
                    "syntactic_success_any": bool(s_ok),
                    "semantic_success_any": bool(se_ok),
                    "plan_found_any": plan_found_for_problem,
                    "last_solver_error": last_solver_error,
                    "last_validator_message": last_validator_msg,
                    "last_error_category": final_category,
                }
            )

            syn_ok += int(s_ok)
            sem_ok += int(se_ok)
            if se_ok:
                continue
            if final_category == "unsolvable":
                plan_not_found += 1
            elif plan_found_for_problem:
                plan_not_valid += 1
            else:
                plan_not_found += 1

        n = len(problem_ids)
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if sem_ok == 0 else total_tokens / sem_ok

        os.makedirs(out_dir, exist_ok=True)
        if per_problem_rows:
            per_problem_path = os.path.join(out_dir, "metrics_pddl_per_problem.csv")
            with open(per_problem_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=per_problem_rows[0].keys())
                writer.writeheader()
                writer.writerows(per_problem_rows)

        stage_rows = [
            {
                "stage": "stage1_final_pddl",
                "syntactic_success_count": syn_ok,
                "semantic_success_count": sem_ok,
                "syntactic_accuracy": syn_ok / n if n else 0.0,
                "semantic_accuracy": sem_ok / n if n else 0.0,
            }
        ]
        self._write_stage_metrics(out_dir, "metrics_pddl_stages.csv", stage_rows)

        metrics = {
            "syntactic_accuracy": syn_ok / n if n else 0.0,
            "semantic_accuracy": sem_ok / n if n else 0.0,
            "syntactic_success_count": syn_ok,
            "semantic_success_count": sem_ok,
            "plan_not_found_count": plan_not_found,
            "plan_not_valid_count": plan_not_valid,
            "total_tokens": total_tokens,
            "tokens_per_valid_plan": tokens_per_valid,
            "n_problems": n,
        }
        metrics.update(self._solver_error_metrics())
        return metrics
