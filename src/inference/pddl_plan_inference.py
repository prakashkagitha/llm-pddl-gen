"""PDDL → Plan two-stage inference pipeline.

Stage 1 generates domain/problem PDDL from natural-language descriptions
using :mod:`prompts/only_pddl_instruction.txt`.  Stage 2 consumes the
generated PDDL alongside the original descriptions and follows
``prompts/pddl_plan_instruction.txt`` to emulate a symbolic planner that
produces an executable plan.  Artefacts from both stages are stored so
downstream analysis can inspect intermediate representations.

The evaluation mirrors :class:`PlanGenInference`, but augments the
per-problem metrics with the Stage 1 PDDL diagnostics (solver feedback,
plan validity, etc.) and aggregates stage-level accuracies.
"""
from __future__ import annotations

import csv
import os
import re
import tempfile
from typing import Dict, List, Tuple

from .base_inference import BaseInference


class PDDLPlanInference(BaseInference):
    """Two-pass generation: draft PDDL then reason out a plan."""

    # ------------------------------------------------------------------ #
    #  Prompt helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pddl_prefix() -> str:
        with open("prompts/only_pddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    @staticmethod
    def _plan_prefix() -> str:
        with open("prompts/pddl_plan_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    def _descs(self, pid: str) -> Tuple[str, str]:
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f:
            dom = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f:
            prob = f.read()
        return dom, prob

    # ------------------------------------------------------------------ #
    #  Generation                                                        #
    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        stage1_prompts: List[str] = []
        contexts: List[Tuple[str, str]] = []
        pddl_prefix = self._pddl_prefix()

        for pid in pids:
            dom_desc, prob_desc = self._descs(pid)
            contexts.append((dom_desc, prob_desc))
            prompt = (
                pddl_prefix
                + "You are an expert PDDL engineer.\n"
                "Derive syntactically correct and mutually consistent domain and "
                "problem files from the natural-language descriptions below.\n\n"
                "### Original domain description\n"
                f"{dom_desc}\n\n"
                "### Original problem description\n"
                f"{prob_desc}\n\n"
                "Wrap the final domain in <domain_file>…</domain_file> and the final "
                "problem in <problem_file>…</problem_file>."
            )
            stage1_prompts.append(self._augment_prompt(prompt))

        stage1_outs = self._tracked_generate(stage1_prompts, self.sampler)
        self._mark_stage_tokens("stage1_pddl")

        stage1_records: List[Dict[str, str]] = []
        for out in stage1_outs:
            stage1_records.append(self._resp2dict(out.outputs[0].text))

        plan_prefix = self._plan_prefix()
        stage2_prompts: List[str] = []
        for (dom_desc, prob_desc), stage1 in zip(contexts, stage1_records):
            stage1_dom = stage1.get("df", "").strip()
            stage1_prob = stage1.get("pf", "").strip()

            prompt = (
                plan_prefix
                + "### Natural-language domain description\n"
                f"{dom_desc}\n\n"
                "### Natural-language problem description\n"
                f"{prob_desc}\n\n"
                "### Draft PDDL domain from Stage 1\n"
                f"{stage1_dom}\n\n"
                "### Draft PDDL problem from Stage 1\n"
                f"{stage1_prob}\n\n"
                "Follow the mandated three-phase reasoning cycle to refine the PDDL "
                "if necessary, emulate graph-search style planning, and return the "
                "final answer using the required <think> and <plan> wrappers."
                "Let's think step by step.\n<think>"
            )
            stage2_prompts.append(self._augment_prompt(prompt))

        stage2_outs = self._tracked_generate(stage2_prompts, self.sampler)
        self._mark_stage_tokens("stage2_plan")

        results: List[List[Dict[str, str]]] = []
        for stage1, out in zip(stage1_records, stage2_outs):
            full = out.outputs[0].text
            stage2_dict = self._resp2dict(full)
            plan_text = self._unwrap(full, "plan").strip()
            think_match = re.findall(r"<think>(.*?)</think>", full, flags=re.DOTALL)
            think_text = think_match[-1].strip() if think_match else ""
            post_think = full.split("</think>", 1)[-1].strip() if "</think>" in full else full.strip()

            final_domain = stage2_dict.get("df") or stage1.get("df", "")
            final_problem = stage2_dict.get("pf") or stage1.get("pf", "")

            cand: Dict[str, str] = {
                "df": final_domain,
                "pf": final_problem,
                "raw": full,
                "plan_text": plan_text,
                "reasoning": think_text,
                "post_think": post_think,
            }

            # Stash Stage 1 artefacts for persistence during evaluation.
            cand["_stage1"] = {
                "df": stage1.get("df", ""),
                "pf": stage1.get("pf", ""),
                "raw": stage1.get("raw", ""),
            }

            results.append([cand])

        return results

    # ------------------------------------------------------------------ #
    #  Solver helper                                                     #
    # ------------------------------------------------------------------ #
    def _solve_once(
        self,
        pid: str,
        domain_src: str,
        problem_src: str,
        *,
        stage_name: str,
    ) -> Tuple[str, str, bool, str]:
        """Run solver + validator on provided PDDL snippets."""
        if not domain_src.strip() or not problem_src.strip():
            return "", "missing PDDL", False, ""

        with tempfile.TemporaryDirectory(prefix="pddl_plan_stage_") as tmp:
            dom_path = os.path.join(tmp, "domain.pddl")
            prob_path = os.path.join(tmp, "problem.pddl")
            with open(dom_path, "w") as f:
                f.write(domain_src)
            with open(prob_path, "w") as f:
                f.write(problem_src)

            plan_txt, solver_err = self._solver.solve_with_error(dom_path, prob_path)
            solver_err = solver_err or ""
            if solver_err:
                self._record_solver_error(solver_err, stage=stage_name)

            if not plan_txt:
                return "", solver_err, False, ""

            plan_path = os.path.join(tmp, "plan.txt")
            with open(plan_path, "w") as f:
                f.write(plan_txt)

            val_ok, val_msg = self._validator.validate_with_error(self.domain, pid, plan_path)
            return plan_txt, solver_err, val_ok, val_msg

    # ------------------------------------------------------------------ #
    #  Persistence                                                       #
    # ------------------------------------------------------------------ #
    def _save_candidate(
        self, out_root: str, pid: str, idx: int, cand: Dict[str, str]
    ) -> Tuple[str, str]:
        stage1 = cand.get("_stage1")
        plan_text = cand.get("plan_text", "")

        # Avoid serialising helper keys directly in BaseInference.
        payload = {k: v for k, v in cand.items() if k not in {"_stage1", "plan_text"}}
        dom_path, prob_path = super()._save_candidate(out_root, pid, idx, payload)

        cand_dir = os.path.join(out_root, pid, f"cand_{idx:02}")
        os.makedirs(cand_dir, exist_ok=True)

        if plan_text:
            plan_lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
            with open(os.path.join(cand_dir, "plan.txt"), "w") as f:
                for line in plan_lines:
                    f.write(f"{line}\n")

        if isinstance(stage1, dict):
            stage_dir = os.path.join(cand_dir, "stage1")
            os.makedirs(stage_dir, exist_ok=True)
            with open(os.path.join(stage_dir, "domain.pddl"), "w") as f:
                f.write(stage1.get("df", ""))
            with open(os.path.join(stage_dir, "problem.pddl"), "w") as f:
                f.write(stage1.get("pf", ""))
            with open(os.path.join(stage_dir, "raw.txt"), "w") as f:
                f.write(stage1.get("raw", ""))

            for extra_key in ("plan", "solver_error", "validator_msg", "syntactic_ok", "semantic_ok"):
                if extra_key in stage1 and stage1[extra_key] not in (None, ""):
                    with open(os.path.join(stage_dir, f"{extra_key}.txt"), "w") as f:
                        f.write(str(stage1[extra_key]))

        return dom_path, prob_path

    # ------------------------------------------------------------------ #
    #  Evaluation                                                        #
    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        os.makedirs(out_dir, exist_ok=True)

        all_cands = self.batch_generate_candidates(problem_ids)

        stage1_syn_total = stage1_sem_total = 0
        plan_syn_total = plan_sem_total = 0
        plan_not_found_total = plan_not_valid_total = 0

        per_problem_rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
            stage1 = cand.get("_stage1") if isinstance(cand.get("_stage1"), dict) else {}

            # --- Stage 1 diagnostics ---------------------------------- #
            stage1_plan = stage1_solver_err = stage1_val_msg = ""
            stage1_syn_ok = stage1_sem_ok = False
            if stage1:
                plan_txt, solver_err, val_ok, val_msg = self._solve_once(
                    pid,
                    stage1.get("df", ""),
                    stage1.get("pf", ""),
                    stage_name="stage1_pddl",
                )
                stage1_plan = plan_txt
                stage1_solver_err = solver_err
                stage1_val_msg = val_msg
                lower_err = solver_err.lower() if solver_err else ""
                stage1_syn_ok = bool(plan_txt)
                if not stage1_syn_ok and lower_err:
                    if "syntax error" not in lower_err and "missing pddl" not in lower_err:
                        stage1_syn_ok = True
                stage1_sem_ok = bool(plan_txt) and val_ok
                stage1.update(
                    {
                        "plan": plan_txt,
                        "solver_error": solver_err,
                        "validator_msg": val_msg,
                        "syntactic_ok": stage1_syn_ok,
                        "semantic_ok": stage1_sem_ok,
                    }
                )

            stage1_syn_total += int(stage1_syn_ok)
            stage1_sem_total += int(stage1_sem_ok)

            # --- Stage 2 plan analysis -------------------------------- #
            raw_plan = cand.get("plan_text", "")
            plan_lines = [ln.strip() for ln in raw_plan.splitlines() if ln.strip()]
            canonical_plan = "\n".join(plan_lines)
            cand["plan_text"] = canonical_plan
            plan_syn_ok = bool(plan_lines)

            cand_dir = os.path.join(out_dir, pid, "cand_00")
            dom_path, prob_path = self._save_candidate(out_dir, pid, 0, cand)

            val_msg = ""
            plan_sem_ok = False
            if plan_syn_ok:
                plan_path = os.path.join(cand_dir, "plan.txt")
                plan_sem_ok, val_msg = self._validator.validate_with_error(
                    self.domain, pid, plan_path
                )
            else:
                val_msg = "Plan empty."

            if val_msg:
                with open(os.path.join(cand_dir, "validator.log"), "w") as f:
                    f.write(val_msg)

            plan_syn_total += int(plan_syn_ok)
            plan_sem_total += int(plan_sem_ok)
            if not plan_syn_ok:
                plan_not_found_total += 1
            elif not plan_sem_ok:
                plan_not_valid_total += 1

            per_problem_rows.append(
                {
                    "problem": pid,
                    "stage1_syntactic_ok": stage1_syn_ok,
                    "stage1_semantic_ok": stage1_sem_ok,
                    "stage1_solver_error": stage1_solver_err,
                    "stage1_validator_message": stage1_val_msg,
                    "plan_length": len(plan_lines),
                    "stage2_syntactic_ok": plan_syn_ok,
                    "stage2_semantic_ok": plan_sem_ok,
                    "stage2_validator_message": val_msg,
                }
            )

        if per_problem_rows:
            with open(os.path.join(out_dir, "metrics_pddl_plan_per_problem.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=per_problem_rows[0].keys())
                writer.writeheader()
                writer.writerows(per_problem_rows)

        n = len(problem_ids) or 1
        stage1_syn_acc = stage1_syn_total / n
        stage1_sem_acc = stage1_sem_total / n
        plan_syn_acc = plan_syn_total / n
        plan_sem_acc = plan_sem_total / n

        stage_rows = [
            {
                "stage": "stage1_pddl",
                "syntactic_success_count": stage1_syn_total,
                "semantic_success_count": stage1_sem_total,
                "syntactic_accuracy": stage1_syn_acc,
                "semantic_accuracy": stage1_sem_acc,
                "cumulative_token_usage": self._stage_token_totals.get("stage1_pddl", 0),
            },
            {
                "stage": "stage2_plan",
                "syntactic_success_count": plan_syn_total,
                "semantic_success_count": plan_sem_total,
                "syntactic_accuracy": plan_syn_acc,
                "semantic_accuracy": plan_sem_acc,
                "cumulative_token_usage": self._stage_token_totals.get("stage2_plan", 0),
            },
        ]
        self._write_stage_metrics(out_dir, "metrics_pddl_plan_stages.csv", stage_rows)

        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if plan_sem_total == 0 else total_tokens / plan_sem_total

        summary_path = os.path.join(out_dir, "metrics_pddl_plan_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage1_syntactic_accuracy",
                    "stage1_semantic_accuracy",
                    "stage2_syntactic_accuracy",
                    "stage2_semantic_accuracy",
                    "n_problems",
                    "plan_not_found_count",
                    "plan_not_valid_count",
                    "total_tokens",
                    "tokens_per_valid_plan",
                    "duplicate_declaration_error_count",
                    "type_or_arity_error_count",
                    "unsolvable_error_count",
                    "miscellaneous_error_count",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "stage1_syntactic_accuracy": stage1_syn_acc,
                    "stage1_semantic_accuracy": stage1_sem_acc,
                    "stage2_syntactic_accuracy": plan_syn_acc,
                    "stage2_semantic_accuracy": plan_sem_acc,
                    "n_problems": len(problem_ids),
                    "plan_not_found_count": plan_not_found_total,
                    "plan_not_valid_count": plan_not_valid_total,
                    "total_tokens": total_tokens,
                    "tokens_per_valid_plan": tokens_per_valid,
                    **self._solver_error_metrics(),
                }
            )

        metrics = {
            "stage1_syntactic_accuracy": stage1_syn_acc,
            "stage1_semantic_accuracy": stage1_sem_acc,
            "stage1_syntactic_success_count": stage1_syn_total,
            "stage1_semantic_success_count": stage1_sem_total,
            "syntactic_accuracy": plan_syn_acc,
            "semantic_accuracy": plan_sem_acc,
            "syntactic_success_count": plan_syn_total,
            "semantic_success_count": plan_sem_total,
            "plan_not_found_count": plan_not_found_total,
            "plan_not_valid_count": plan_not_valid_total,
            "total_tokens": total_tokens,
            "tokens_per_valid_plan": tokens_per_valid,
            "n_problems": len(problem_ids),
        }
        metrics.update(self._solver_error_metrics())
        return metrics

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "PDDLPlanInference constructs prompts inside batch_generate_candidates."
        )
