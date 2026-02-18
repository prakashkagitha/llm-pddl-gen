"""
Three-stage PDDL → PDDL → PDDL pipeline without exposing solver feedback.

All solver diagnostics are still collected for metrics, but revision
stages only see the previous artefacts. This allows analysing how much
solver feedback contributes relative to pure self-revision.
"""
from __future__ import annotations

import csv
import os
import tempfile
from typing import Dict, List, Tuple

from .pddl_knowledge_inference import PDDLKnowledgeInference


class PDDLPDDLPDDLNoFBInference(PDDLKnowledgeInference):
    """Three-step direct PDDL refinement pipeline without solver feedback."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stage_cache: Dict[str, Dict[str, Dict[str, object]]] = {}

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
        """Run solver + validator, returning plan/solver_err/semantic_ok/validator_msg."""
        with tempfile.TemporaryDirectory(prefix="pddl_stage_") as tmp:
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

    @staticmethod
    def _revision_prompt(
        base_prompt: str,
        previous_domain: str,
        previous_problem: str,
        *,
        guidance: str,
    ) -> str:
        return (
            base_prompt
            + "\n\n------ Previous attempt start ------\n"
            + "[DOMAIN FILE]\n" + previous_domain + "\n\n"
            + "[PROBLEM FILE]\n" + previous_problem + "\n"
            + "------ Previous attempt end --------\n\n"
            + guidance
            + "\n\nOutput only the revised domain wrapped in <domain_file>…</domain_file> and the revised problem wrapped in <problem_file>…</problem_file>."
        )

    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        contexts: Dict[str, Tuple[str, str]] = {}
        stage1_prompts: List[str] = []
        for pid in pids:
            dom_desc, prob_desc = self._descs(pid)
            contexts[pid] = (dom_desc, prob_desc)
            prompt = (
                self._base_prompt(dom_desc, prob_desc)
                + "\n\nWrap the domain inside <domain_file>…</domain_file> and the problem inside "
                "<problem_file>…</problem_file>."
            )
            stage1_prompts.append(self._augment_prompt(prompt))

        stage1_outs = self._tracked_generate(stage1_prompts, self.sampler)
        self._mark_stage_tokens("stage1_initial_pddl")

        stage1_records: Dict[str, Dict[str, object]] = {}
        stage2_prompts: List[str] = []
        for pid, out in zip(pids, stage1_outs):
            stage1_dict = self._resp2dict(out.outputs[0].text)
            plan_str, solver_err, sem_ok, val_msg = self._solve_once(
                pid,
                stage1_dict.get("df", ""),
                stage1_dict.get("pf", ""),
                stage_name="stage1_initial_pddl",
            )
            solver_err_lower = solver_err.lower() if solver_err else ""
            syn_ok = bool(plan_str) or ("syntax error" not in solver_err_lower)

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
            stage1_records[pid] = stage1_record

            dom_desc, prob_desc = contexts[pid]
            base_prompt = self._base_prompt(dom_desc, prob_desc)
            guidance = (
                "No solver diagnostics are available. Carefully inspect the prior PDDL to resolve any latent issues."
            )
            prompt = self._revision_prompt(
                base_prompt,
                stage1_record["df"],
                stage1_record["pf"],
                guidance=guidance,
            )
            stage2_prompts.append(self._augment_prompt(prompt))

        stage2_outs = self._tracked_generate(stage2_prompts, self.sampler)
        self._mark_stage_tokens("stage2_revision_pddl")

        stage2_records: Dict[str, Dict[str, object]] = {}
        stage3_prompts: List[str] = []
        for pid, out in zip(pids, stage2_outs):
            stage2_dict = self._resp2dict(out.outputs[0].text)
            plan_str, solver_err, sem_ok, val_msg = self._solve_once(
                pid,
                stage2_dict.get("df", ""),
                stage2_dict.get("pf", ""),
                stage_name="stage2_revision_pddl",
            )
            solver_err_lower = solver_err.lower() if solver_err else ""
            syn_ok = bool(plan_str) or ("syntax error" not in solver_err_lower)

            stage2_record = {
                "df": stage2_dict.get("df", ""),
                "pf": stage2_dict.get("pf", ""),
                "raw": stage2_dict.get("raw", ""),
                "plan": plan_str,
                "solver_error": solver_err,
                "validator_msg": val_msg,
                "syntactic_ok": syn_ok,
                "semantic_ok": sem_ok,
            }
            stage2_records[pid] = stage2_record

            dom_desc, prob_desc = contexts[pid]
            base_prompt = self._base_prompt(dom_desc, prob_desc)
            guidance = (
                "Produce a polished final version of the PDDL. Rely on the previous specification and your own checks;"
                " no solver feedback is provided."
            )
            prompt = (
                base_prompt
                + "\n\n### Previous domain.pddl\n```pddl\n"
                + stage2_record["df"]
                + "\n```\n\n"
                "### Previous problem.pddl\n```pddl\n"
                + stage2_record["pf"]
                + "\n```\n\n"
                + guidance
                + "\n\nWrap the refined domain in <domain_file>…</domain_file> and the refined problem in <problem_file>…</problem_file>."
            )
            stage3_prompts.append(self._augment_prompt(prompt))

        final_outs = self._tracked_generate(stage3_prompts, self.sampler)
        self._mark_stage_tokens("stage3_final_pddl")

        results: List[List[Dict[str, str]]] = []
        self._stage_cache = {}
        for pid, out in zip(pids, final_outs):
            cand = self._resp2dict(out.outputs[0].text)
            results.append([cand])
            self._stage_cache[pid] = {
                "stage1": stage1_records.get(pid, {}),
                "stage2": stage2_records.get(pid, {}),
            }

        return results

    # ------------------------------------------------------------------ #
    def _save_candidate(
        self, out_root: str, pid: str, idx: int, cand: Dict[str, str]
    ) -> Tuple[str, str]:
        stage_data = self._stage_cache.get(pid, {})
        dom_path, prob_path = super()._save_candidate(out_root, pid, idx, cand)

        cand_dir = os.path.join(out_root, pid, f"cand_{idx:02}")

        stage1 = stage_data.get("stage1")
        if isinstance(stage1, dict) and stage1:
            stage1_dir = os.path.join(cand_dir, "stage1_initial")
            os.makedirs(stage1_dir, exist_ok=True)
            with open(os.path.join(stage1_dir, "domain.pddl"), "w") as f:
                f.write(str(stage1.get("df", "")))
            with open(os.path.join(stage1_dir, "problem.pddl"), "w") as f:
                f.write(str(stage1.get("pf", "")))
            with open(os.path.join(stage1_dir, "raw.txt"), "w") as f:
                f.write(str(stage1.get("raw", "")))
            with open(os.path.join(stage1_dir, "metrics.txt"), "w") as f:
                f.write(
                    "syntactic_ok: {}\nsemantic_ok: {}\n".format(
                        stage1.get("syntactic_ok", False),
                        stage1.get("semantic_ok", False),
                    )
                )
            if stage1.get("plan"):
                with open(os.path.join(stage1_dir, "plan.txt"), "w") as f:
                    f.write(str(stage1.get("plan", "")))
            if stage1.get("solver_error"):
                with open(os.path.join(stage1_dir, "solver_error.txt"), "w") as f:
                    f.write(str(stage1.get("solver_error", "")))
            if stage1.get("validator_msg"):
                with open(os.path.join(stage1_dir, "validator.log"), "w") as f:
                    f.write(str(stage1.get("validator_msg", "")))

        stage2 = stage_data.get("stage2")
        if isinstance(stage2, dict) and stage2:
            stage2_dir = os.path.join(cand_dir, "stage2_revision")
            os.makedirs(stage2_dir, exist_ok=True)
            with open(os.path.join(stage2_dir, "domain.pddl"), "w") as f:
                f.write(str(stage2.get("df", "")))
            with open(os.path.join(stage2_dir, "problem.pddl"), "w") as f:
                f.write(str(stage2.get("pf", "")))
            with open(os.path.join(stage2_dir, "raw.txt"), "w") as f:
                f.write(str(stage2.get("raw", "")))
            with open(os.path.join(stage2_dir, "metrics.txt"), "w") as f:
                f.write(
                    "syntactic_ok: {}\nsemantic_ok: {}\n".format(
                        stage2.get("syntactic_ok", False),
                        stage2.get("semantic_ok", False),
                    )
                )
            if stage2.get("plan"):
                with open(os.path.join(stage2_dir, "plan.txt"), "w") as f:
                    f.write(str(stage2.get("plan", "")))
            if stage2.get("solver_error"):
                with open(os.path.join(stage2_dir, "solver_error.txt"), "w") as f:
                    f.write(str(stage2.get("solver_error", "")))
            if stage2.get("validator_msg"):
                with open(os.path.join(stage2_dir, "validator.log"), "w") as f:
                    f.write(str(stage2.get("validator_msg", "")))

        return dom_path, prob_path

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        stage1_syn = stage1_sem = 0
        stage2_syn = stage2_sem = 0
        final_syn = final_sem = 0
        stage1_plan_not_found = stage1_plan_not_valid = 0
        stage2_plan_not_found = stage2_plan_not_valid = 0
        plan_not_found_total = plan_not_valid_total = 0
        rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
            stages = self._stage_cache.get(pid, {})
            s1 = stages.get("stage1", {})
            s2 = stages.get("stage2", {})

            stage1_syn += int(bool(s1.get("syntactic_ok")))
            stage1_sem += int(bool(s1.get("semantic_ok")))
            stage1_category = self._categorize_solver_error(str(s1.get("solver_error", "")))
            stage1_plan_found = bool(s1.get("plan"))
            if not bool(s1.get("semantic_ok")):
                if stage1_category == "unsolvable":
                    stage1_plan_not_found += 1
                elif stage1_plan_found:
                    stage1_plan_not_valid += 1
                else:
                    stage1_plan_not_found += 1

            stage2_syn += int(bool(s2.get("syntactic_ok")))
            stage2_sem += int(bool(s2.get("semantic_ok")))
            stage2_category = self._categorize_solver_error(str(s2.get("solver_error", "")))
            stage2_plan_found = bool(s2.get("plan"))
            if not bool(s2.get("semantic_ok")):
                if stage2_category == "unsolvable":
                    stage2_plan_not_found += 1
                elif stage2_plan_found:
                    stage2_plan_not_valid += 1
                else:
                    stage2_plan_not_found += 1

            syn_ok, sem_ok, sol_err, val_msg, plan_found, category = self._check_candidate(
                pid, 0, cand, out_dir, stage_name="stage3_final_pddl"
            )
            final_syn += int(syn_ok)
            final_sem += int(sem_ok)
            if not sem_ok:
                if category == "unsolvable":
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

            rows.append(
                {
                    "problem": pid,
                    "stage1_syntactic_ok": bool(s1.get("syntactic_ok")),
                    "stage1_semantic_ok": bool(s1.get("semantic_ok")),
                    "stage1_error_category": stage1_category,
                    "stage2_syntactic_ok": bool(s2.get("syntactic_ok")),
                    "stage2_semantic_ok": bool(s2.get("semantic_ok")),
                    "stage2_error_category": stage2_category,
                    "final_syntactic_ok": syn_ok,
                    "final_semantic_ok": sem_ok,
                    "final_error_category": category,
                }
            )

        if rows:
            with open(os.path.join(out_dir, "metrics_pddl_pddl_pddl_nofb_per_problem.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        stage1_syn_acc = stage1_syn / len(problem_ids) if problem_ids else 0.0
        stage1_sem_acc = stage1_sem / len(problem_ids) if problem_ids else 0.0
        stage2_syn_acc = stage2_syn / len(problem_ids) if problem_ids else 0.0
        stage2_sem_acc = stage2_sem / len(problem_ids) if problem_ids else 0.0
        final_syn_acc = final_syn / len(problem_ids) if problem_ids else 0.0
        final_sem_acc = final_sem / len(problem_ids) if problem_ids else 0.0
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if final_sem == 0 else total_tokens / final_sem
        stage1_tokens_total = self._stage_token_totals.get("stage1_initial_pddl", total_tokens)
        stage2_tokens_total = self._stage_token_totals.get("stage2_revision_pddl", total_tokens)
        stage3_tokens_total = self._stage_token_totals.get("stage3_final_pddl", total_tokens)
        stage1_tokens_per_valid = (
            float("nan") if stage1_sem == 0 else stage1_tokens_total / stage1_sem
        )
        stage2_tokens_per_valid = (
            float("nan") if stage2_sem == 0 else stage2_tokens_total / stage2_sem
        )

        stage1_errors = self._solver_error_metrics("stage1_initial_pddl")
        stage2_errors = self._solver_error_metrics("stage2_revision_pddl")
        stage3_errors = self._solver_error_metrics("stage3_final_pddl")

        stage_rows = [
            {
                "stage": "stage1_initial_pddl",
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
                "stage": "stage2_revision_pddl",
                "syntactic_success_count": stage2_syn,
                "semantic_success_count": stage2_sem,
                "syntactic_accuracy": stage2_syn_acc,
                "semantic_accuracy": stage2_sem_acc,
                "plan_not_found_count": stage2_plan_not_found,
                "plan_not_valid_count": stage2_plan_not_valid,
                "n_problems": len(problem_ids),
                "total_tokens": stage2_tokens_total,
                "tokens_per_valid_plan": stage2_tokens_per_valid,
                "stage1_syntactic_accuracy": stage1_syn_acc,
                "stage1_semantic_accuracy": stage1_sem_acc,
                "stage2_syntactic_accuracy": stage2_syn_acc,
                "stage2_semantic_accuracy": stage2_sem_acc,
                **stage2_errors,
            },
            {
                "stage": "stage3_final_pddl",
                "syntactic_success_count": final_syn,
                "semantic_success_count": final_sem,
                "syntactic_accuracy": final_syn_acc,
                "semantic_accuracy": final_sem_acc,
                "plan_not_found_count": plan_not_found_total,
                "plan_not_valid_count": plan_not_valid_total,
                "n_problems": len(problem_ids),
                "total_tokens": stage3_tokens_total,
                "tokens_per_valid_plan": tokens_per_valid,
                "stage1_syntactic_accuracy": stage1_syn_acc,
                "stage1_semantic_accuracy": stage1_sem_acc,
                "stage2_syntactic_accuracy": stage2_syn_acc,
                "stage2_semantic_accuracy": stage2_sem_acc,
                "final_syntactic_accuracy": final_syn_acc,
                "final_semantic_accuracy": final_sem_acc,
                **stage3_errors,
            },
        ]
        self._write_stage_metrics(out_dir, "metrics_pddl_pddl_pddl_nofb_stages.csv", stage_rows)

        with open(os.path.join(out_dir, "metrics_pddl_pddl_pddl_nofb_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage1_syntactic_accuracy",
                    "stage1_semantic_accuracy",
                    "stage2_syntactic_accuracy",
                    "stage2_semantic_accuracy",
                    "final_syntactic_accuracy",
                    "final_semantic_accuracy",
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
                    "stage2_syntactic_accuracy": stage2_syn_acc,
                    "stage2_semantic_accuracy": stage2_sem_acc,
                    "final_syntactic_accuracy": final_syn_acc,
                    "final_semantic_accuracy": final_sem_acc,
                    "n_problems": len(problem_ids),
                    "plan_not_found_count": plan_not_found_total,
                    "plan_not_valid_count": plan_not_valid_total,
                    "total_tokens": total_tokens,
                    "tokens_per_valid_plan": tokens_per_valid,
                    **self._solver_error_metrics(),
                }
            )

        metrics = {
            "syntactic_accuracy": final_syn_acc,
            "semantic_accuracy": final_sem_acc,
            "syntactic_success_count": final_syn,
            "semantic_success_count": final_sem,
            "plan_not_found_count": plan_not_found_total,
            "plan_not_valid_count": plan_not_valid_total,
            "total_tokens": total_tokens,
            "tokens_per_valid_plan": tokens_per_valid,
            "n_problems": len(problem_ids),
            "stage1_syntactic_accuracy": stage1_syn_acc,
            "stage1_semantic_accuracy": stage1_sem_acc,
            "stage2_syntactic_accuracy": stage2_syn_acc,
            "stage2_semantic_accuracy": stage2_sem_acc,
        }
        metrics.update(self._solver_error_metrics())
        return metrics

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "PDDLPDDLPDDLNoFBInference constructs prompts inside batch_generate_candidates."
        )
