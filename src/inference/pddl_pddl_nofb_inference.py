"""
Two-stage PDDL → PDDL pipeline without solver feedback for revisions.

Stage 1 generates domain/problem PDDL directly from natural language.
If the solver fails to find a plan, Stage 2 performs a revision, but the
model only sees the previous artefacts — no solver error messages or
plans. All intermediate artefacts and metrics mirror the feedback-based
pipeline so downstream tooling can compare behaviours directly.
"""
from __future__ import annotations

import csv
import os
import tempfile
from typing import Dict, List, Optional, Tuple

from .pddl_knowledge_inference import PDDLKnowledgeInference


class PDDLPDDLNoFBInference(PDDLKnowledgeInference):
    """Direct PDDL generation followed by a solver-feedback-free revision."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stage1_cache: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------ #
    def _base_prompt(self, dom_desc: str, prob_desc: str) -> str:
        return (
            self._prefix()
            + "Domain description:\n"
            + dom_desc
            + "\n\nProblem description:\n"
            + prob_desc
            + "\nWrite the domain and problem files in minimal PDDL."
        )

    # ------------------------------------------------------------------ #
    def _solve_once(
        self,
        pid: str,
        domain_src: str,
        problem_src: str,
        *,
        stage_name: str,
    ) -> Tuple[str, str, bool, str]:
        """Run the solver and validator, returning diagnostics for metrics."""
        with tempfile.TemporaryDirectory(prefix="pddl_stage1_") as tmp:
            dom_path = os.path.join(tmp, "domain.pddl")
            prob_path = os.path.join(tmp, "problem.pddl")
            with open(dom_path, "w") as f:
                f.write(domain_src)
            with open(prob_path, "w") as f:
                f.write(problem_src)

            plan, solver_err = self._solver.solve_with_error(dom_path, prob_path)
            val_ok = False
            val_msg = ""
            plan_str = ""

            if plan:
                plan_path = os.path.join(tmp, "plan.txt")
                with open(plan_path, "w") as f:
                    f.write(str(plan))
                val_ok, val_msg = self._validator.validate_with_error(
                    self.domain, pid, plan_path
                )
                plan_str = str(plan)

            solver_err = solver_err or ""
            if solver_err:
                self._record_solver_error(solver_err, stage=stage_name)
            return plan_str, solver_err, val_ok, val_msg

    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        contexts: List[Tuple[str, str]] = []
        stage1_prompts: List[str] = []
        for pid in pids:
            dom_desc, prob_desc = self._descs(pid)
            contexts.append((dom_desc, prob_desc))
            prompt = (
                self._base_prompt(dom_desc, prob_desc)
                + "\n\nWrap the domain inside <domain_file>…</domain_file> and the problem inside "
                "<problem_file>…</problem_file>."
            )
            stage1_prompts.append(self._augment_prompt(prompt))
        stage1_outs = self._tracked_generate(stage1_prompts, self.sampler)
        self._mark_stage_tokens("stage1_initial")

        self._stage1_cache = {}
        final_candidates: List[Optional[Dict[str, str]]] = [None] * len(pids)
        revision_payloads: List[Tuple[int, str]] = []
        revision_prompts: List[str] = []

        for idx, (pid, out, ctx) in enumerate(zip(pids, stage1_outs, contexts)):
            stage1_dict = self._resp2dict(out.outputs[0].text)
            plan_str, solver_err, sem_ok, val_msg = self._solve_once(
                pid,
                stage1_dict.get("df", ""),
                stage1_dict.get("pf", ""),
                stage_name="stage1_initial",
            )
            syn_ok = bool(plan_str) or "syntax error" not in solver_err.lower()
            stage1_record = {
                "df": stage1_dict.get("df", ""),
                "pf": stage1_dict.get("pf", ""),
                "raw": stage1_dict.get("raw", ""),
                "plan": plan_str,
                "solver_error": solver_err,
                "validator_msg": val_msg,
                "syntactic_ok": syn_ok,
                "semantic_ok": sem_ok,
            }
            self._stage1_cache[pid] = stage1_record

            if plan_str:
                stage1_dict["_stage1"] = stage1_record
                final_candidates[idx] = stage1_dict
                continue

            dom_desc, prob_desc = ctx
            base_prompt = self._base_prompt(dom_desc, prob_desc)
            prompt = (
                base_prompt
                + "\n\nTo aid the revision, the original natural-language descriptions are repeated above."
                "\n------ Previous attempt start ------\n"
                + "[DOMAIN FILE]\n" + stage1_record["df"] + "\n\n"
                + "[PROBLEM FILE]\n" + stage1_record["pf"] + "\n"
                + "------ Previous attempt end --------\n\n"
                + "No solver feedback is available. Carefully inspect the previous PDDL and produce a corrected domain"
                  " and problem. Output only the revised domain wrapped in <domain_file>…</domain_file> and the revised"
                  " problem wrapped in <problem_file>…</problem_file>."
            )
            revision_prompts.append(self._augment_prompt(prompt))
            revision_payloads.append((idx, pid))

        if revision_prompts:
            revision_outs = self._tracked_generate(revision_prompts, self.sampler)
            self._mark_stage_tokens("stage2_final")
            for (idx, pid), out in zip(revision_payloads, revision_outs):
                revised = self._resp2dict(out.outputs[0].text)
                revised["_stage1"] = self._stage1_cache[pid]
                final_candidates[idx] = revised

        for idx, cand in enumerate(final_candidates):
            if cand is None:
                fallback = self._resp2dict(stage1_outs[idx].outputs[0].text)
                fallback["_stage1"] = self._stage1_cache[pids[idx]]
                final_candidates[idx] = fallback

        if "stage2_final" not in self._stage_token_totals:
            self._mark_stage_tokens("stage2_final")

        return [[cand] for cand in final_candidates]  # type: ignore[list-item]

    # ------------------------------------------------------------------ #
    def _save_candidate(
        self, out_root: str, pid: str, idx: int, cand: Dict[str, str]
    ):
        stage1 = cand.get("_stage1")
        cand_to_save = {k: v for k, v in cand.items() if k != "_stage1"}
        dom_path, prob_path = super()._save_candidate(out_root, pid, idx, cand_to_save)

        if isinstance(stage1, dict):
            cand_dir = os.path.join(out_root, pid, f"cand_{idx:02}")
            stage_dir = os.path.join(cand_dir, "stage_initial")
            os.makedirs(stage_dir, exist_ok=True)

            with open(os.path.join(stage_dir, "domain.pddl"), "w") as f:
                f.write(stage1.get("df", ""))
            with open(os.path.join(stage_dir, "problem.pddl"), "w") as f:
                f.write(stage1.get("pf", ""))
            with open(os.path.join(stage_dir, "raw.txt"), "w") as f:
                f.write(stage1.get("raw", ""))

            solver_err = stage1.get("solver_error", "")
            if solver_err:
                with open(os.path.join(stage_dir, "solver_error.txt"), "w") as f:
                    f.write(str(solver_err))

            plan_text = stage1.get("plan", "")
            if plan_text:
                with open(os.path.join(stage_dir, "plan.txt"), "w") as f:
                    f.write(str(plan_text))

            val_msg = stage1.get("validator_msg", "")
            if val_msg:
                with open(os.path.join(stage_dir, "validator.log"), "w") as f:
                    f.write(str(val_msg))

            with open(os.path.join(stage_dir, "metrics.txt"), "w") as f:
                f.write(
                    "syntactic_ok: {}\nsemantic_ok: {}\n".format(
                        stage1.get("syntactic_ok", False),
                        stage1.get("semantic_ok", False),
                    )
                )

        return dom_path, prob_path

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        stage1_syn = stage1_sem = stage2_syn = stage2_sem = 0
        stage1_plan_not_found = stage1_plan_not_valid = 0
        plan_not_found_total = plan_not_valid_total = 0
        detailed_rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
            stage1 = self._stage1_cache.get(pid, {})
            stage1_syn += int(bool(stage1.get("syntactic_ok")))
            stage1_sem += int(bool(stage1.get("semantic_ok")))
            stage1_plan_found = bool(stage1.get("plan"))
            stage1_sem_ok = bool(stage1.get("semantic_ok"))
            stage1_category = self._categorize_solver_error(stage1.get("solver_error", ""))
            if not stage1_sem_ok:
                if stage1_category == "unsolvable":
                    stage1_plan_not_found += 1
                elif stage1_plan_found:
                    stage1_plan_not_valid += 1
                else:
                    stage1_plan_not_found += 1

            cand["_stage1"] = stage1
            syn_ok, sem_ok, sol_err, val_msg, plan_found, stage2_category = self._check_candidate(
                pid, 0, cand, out_dir, stage_name="stage2_final"
            )
            stage2_syn += int(syn_ok)
            stage2_sem += int(sem_ok)
            if not sem_ok:
                if stage2_category == "unsolvable":
                    plan_not_found_total += 1
                elif plan_found:
                    plan_not_valid_total += 1
                else:
                    plan_not_found_total += 1

            cand_dir = os.path.join(out_dir, pid, "cand_00")
            if sol_err:
                with open(os.path.join(cand_dir, "solver_error.txt"), "w") as f:
                    f.write(sol_err)
            if val_msg:
                with open(os.path.join(cand_dir, "validator.log"), "w") as f:
                    f.write(val_msg)

            detailed_rows.append(
                {
                    "problem": pid,
                    "stage1_syntactic_ok": bool(stage1.get("syntactic_ok")),
                    "stage1_semantic_ok": bool(stage1.get("semantic_ok")),
                    "stage1_solver_error": stage1.get("solver_error", ""),
                    "stage1_validator_msg": stage1.get("validator_msg", ""),
                    "stage1_error_category": stage1_category,
                    "stage2_syntactic_ok": syn_ok,
                    "stage2_semantic_ok": sem_ok,
                    "stage2_solver_error": sol_err,
                    "stage2_validator_msg": val_msg,
                    "stage2_error_category": stage2_category,
                }
            )

        stage1_syn_acc = stage1_syn / len(problem_ids) if problem_ids else 0.0
        stage1_sem_acc = stage1_sem / len(problem_ids) if problem_ids else 0.0
        stage2_syn_acc = stage2_syn / len(problem_ids) if problem_ids else 0.0
        stage2_sem_acc = stage2_sem / len(problem_ids) if problem_ids else 0.0
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if stage2_sem == 0 else total_tokens / stage2_sem
        stage1_tokens_total = self._stage_token_totals.get("stage1_initial", total_tokens)
        stage2_tokens_total = self._stage_token_totals.get("stage2_final", total_tokens)
        stage1_tokens_per_valid = (
            float("nan") if stage1_sem == 0 else stage1_tokens_total / stage1_sem
        )

        stage1_errors = self._solver_error_metrics("stage1_initial")
        stage2_errors = self._solver_error_metrics("stage2_final")

        os.makedirs(out_dir, exist_ok=True)
        per_problem_path = os.path.join(out_dir, "metrics_pddl_pddl_nofb_per_problem.csv")
        if detailed_rows:
            with open(per_problem_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=detailed_rows[0].keys())
                writer.writeheader()
                writer.writerows(detailed_rows)

        stage_rows = [
            {
                "stage": "stage1_initial",
                "syntactic_success_count": stage1_syn,
                "semantic_success_count": stage1_sem,
                "syntactic_accuracy": stage1_syn_acc,
                "semantic_accuracy": stage1_sem_acc,
                "plan_not_found_count": stage1_plan_not_found,
                "plan_not_valid_count": stage1_plan_not_valid,
                "n_problems": len(problem_ids),
                "total_tokens": stage1_tokens_total,
                "tokens_per_valid_plan": stage1_tokens_per_valid,
                "stage1_syntactic_accuracy": stage1_syn_acc,
                "stage1_semantic_accuracy": stage1_sem_acc,
                **stage1_errors,
            },
            {
                "stage": "stage2_final",
                "syntactic_success_count": stage2_syn,
                "semantic_success_count": stage2_sem,
                "syntactic_accuracy": stage2_syn_acc,
                "semantic_accuracy": stage2_sem_acc,
                "plan_not_found_count": plan_not_found_total,
                "plan_not_valid_count": plan_not_valid_total,
                "n_problems": len(problem_ids),
                "total_tokens": stage2_tokens_total,
                "tokens_per_valid_plan": tokens_per_valid,
                "stage1_syntactic_accuracy": stage1_syn_acc,
                "stage1_semantic_accuracy": stage1_sem_acc,
                **stage2_errors,
            },
        ]
        self._write_stage_metrics(out_dir, "metrics_pddl_pddl_nofb_stages.csv", stage_rows)

        summary_path = os.path.join(out_dir, "metrics_pddl_pddl_nofb_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "syntactic_accuracy",
                    "semantic_accuracy",
                    "n_problems",
                    "plan_not_found_count",
                    "plan_not_valid_count",
                    "total_tokens",
                    "tokens_per_valid_plan",
                    "stage1_syntactic_accuracy",
                    "stage1_semantic_accuracy",
                    "duplicate_declaration_error_count",
                    "type_or_arity_error_count",
                    "unsolvable_error_count",
                    "miscellaneous_error_count",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "syntactic_accuracy": stage2_syn_acc,
                    "semantic_accuracy": stage2_sem_acc,
                    "n_problems": len(problem_ids),
                    "plan_not_found_count": plan_not_found_total,
                    "plan_not_valid_count": plan_not_valid_total,
                    "total_tokens": total_tokens,
                    "tokens_per_valid_plan": tokens_per_valid,
                    "stage1_syntactic_accuracy": stage1_syn_acc,
                    "stage1_semantic_accuracy": stage1_sem_acc,
                    **self._solver_error_metrics(),
                }
            )

        summary = {
            "syntactic_accuracy": stage2_syn_acc,
            "semantic_accuracy": stage2_sem_acc,
            "syntactic_success_count": stage2_syn,
            "semantic_success_count": stage2_sem,
            "plan_not_found_count": plan_not_found_total,
            "plan_not_valid_count": plan_not_valid_total,
            "total_tokens": total_tokens,
            "tokens_per_valid_plan": tokens_per_valid,
            "n_problems": len(problem_ids),
            "stage1_syntactic_accuracy": stage1_syn_acc,
            "stage1_semantic_accuracy": stage1_sem_acc,
        }
        summary.update(self._solver_error_metrics())
        return summary
