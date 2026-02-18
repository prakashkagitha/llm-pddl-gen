"""
NL → NL → PDDL pipeline with LLM-guided revision.

Stage 1 produces a natural-language planning specification. Stage 2
collects feedback plus actionable repair advice from an LLM reviewer.
Stage 3 rewrites the NL specification using that guidance, and Stage 4
converts the refined prose into PDDL domain and problem files.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

from .base_inference import BaseInference


class NLNLPDDLInference(BaseInference):
    """Three-stage (NL, NL, PDDL) pipeline with an explicit review loop."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stage_cache: Dict[str, Dict[str, Dict[str, object]]] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _nl_prefix() -> str:
        with open("prompts/nl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    @staticmethod
    def _pddl_prefix() -> str:
        with open("prompts/pddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    def _descs(self, pid: str) -> Tuple[str, str]:
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f:
            dom = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f:
            prob = f.read()
        return dom, prob

    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        dom_texts: List[str] = []
        prob_texts: List[str] = []
        stage1_prompts: List[str] = []
        for pid in pids:
            dom, prob = self._descs(pid)
            dom_texts.append(dom)
            prob_texts.append(prob)
            prompt = (
                self._nl_prefix()
                + "### Domain description\n"
                + f"{dom}\n\n"
                + "### Problem description\n"
                + f"{prob}\n\n"
                + "Produce the full natural-language planning specification. Wrap it in "
                "<nl_summary>…</nl_summary> and start reasoning with <think>."
            )
            stage1_prompts.append(self._augment_prompt(prompt))

        stage1_outs = self._tracked_generate(stage1_prompts, self.sampler)
        stage1_infos: List[Dict[str, object]] = []
        for out in stage1_outs:
            full = out.outputs[0].text
            summary = self._unwrap(full, "nl_summary")
            if not summary:
                summary = full.split("</think>", 1)[-1].strip()
            stage1_infos.append(
                {
                    "summary": summary,
                    "raw": full.split("</think>", 1)[-1].strip(),
                }
            )

        stage2_feedback_prompts: List[str] = []
        for dom_desc, prob_desc, info in zip(dom_texts, prob_texts, stage1_infos):
            summary = str(info.get("summary", ""))
            prompt = (
                "You are auditing a natural-language planning specification. "
                "Reason carefully about fidelity to the task, missing details, and ambiguous phrasing. "
                "Perform a thorough internal analysis before replying, then report only a concise external summary.\n\n"
                "### Original domain description\n"
                f"{dom_desc}\n\n"
                "### Original problem description\n"
                f"{prob_desc}\n\n"
                "### Draft planning specification\n"
                f"{summary}\n\n"
                "Respond with\n"
                "<llm_feedback>Summarise the critical issues you found and give specific repair instructions. Keep it succinct and actionable.</llm_feedback>\n"
                "Do not include any other XML tags in your final answer."
            )
            stage2_feedback_prompts.append(self._augment_prompt(prompt))

        stage2_feedback_outs = self._tracked_generate_token_free(
            stage2_feedback_prompts, self.sampler
        )
        stage2_feedback_infos: List[Dict[str, object]] = []
        for out in stage2_feedback_outs:
            full = out.outputs[0].text
            tail = full.split("</think>", 1)[-1].strip()
            feedback = (self._unwrap(full, "llm_feedback") or "").strip()
            if not feedback:
                feedback = tail
            stage2_feedback_infos.append(
                {
                    "feedback": feedback,
                    "repair_advice": feedback,
                    "raw": tail,
                }
            )

        stage3_prompts: List[str] = []
        for dom_desc, prob_desc, stage1_info, feedback_info in zip(
            dom_texts, prob_texts, stage1_infos, stage2_feedback_infos
        ):
            summary = str(stage1_info.get("summary", ""))
            feedback_txt = str(feedback_info.get("feedback", "")).strip()
            if not feedback_txt:
                feedback_txt = (
                    "The reviewer found no major mistakes but requested improved clarity and completeness."
                )

            prompt = (
                self._nl_prefix()
                + "Revise the planning specification using the reviewer feedback. Maintain fidelity to the original task details.\n\n"
                "### Original domain description\n"
                f"{dom_desc}\n\n"
                "### Original problem description\n"
                f"{prob_desc}\n\n"
                "### Review summary\n"
                f"{feedback_txt}\n\n"
                "### Previous planning specification\n"
                f"{summary}\n\n"
                "Output the updated specification inside <nl_summary>…</nl_summary>."
            )
            stage3_prompts.append(self._augment_prompt(prompt))

        stage3_outs = self._tracked_generate(stage3_prompts, self.sampler)
        stage3_infos: List[Dict[str, object]] = []
        for out in stage3_outs:
            full = out.outputs[0].text
            summary = self._unwrap(full, "nl_summary")
            if not summary:
                summary = full.split("</think>", 1)[-1].strip()
            stage3_infos.append(
                {
                    "summary": summary,
                    "raw": full.split("</think>", 1)[-1].strip(),
                }
            )

        stage4_prompts: List[str] = []
        for dom_desc, prob_desc, info in zip(dom_texts, prob_texts, stage3_infos):
            summary = str(info.get("summary", ""))
            prompt = (
                self._pddl_prefix()
                + "Use the refined planning specification to write PDDL domain and problem files.\n\n"
                "### Original domain description\n"
                f"{dom_desc}\n\n"
                "### Original problem description\n"
                f"{prob_desc}\n\n"
                "### Refined planning specification\n"
                f"{summary}\n\n"
                "Wrap the domain file in <domain_file>…</domain_file> and the problem file in <problem_file>…</problem_file>."
            )
            stage4_prompts.append(self._augment_prompt(prompt))

        final_outs = self._tracked_generate(stage4_prompts, self.sampler)

        results: List[List[Dict[str, str]]] = []
        self._stage_cache = {}
        for pid, out, s1, s2_feedback, s3 in zip(
            pids, final_outs, stage1_infos, stage2_feedback_infos, stage3_infos
        ):
            cand = self._resp2dict(out.outputs[0].text, summary=str(s3.get("summary", "")))
            results.append([cand])
            self._stage_cache[pid] = {
                "stage1": s1,
                "stage2_feedback": s2_feedback,
                "stage2_nl": s3,
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
        if stage1:
            stage_dir = os.path.join(cand_dir, "stage1_nl_initial")
            os.makedirs(stage_dir, exist_ok=True)
            with open(os.path.join(stage_dir, "summary.txt"), "w") as f:
                f.write(str(stage1.get("summary", "")))
            with open(os.path.join(stage_dir, "raw.txt"), "w") as f:
                f.write(str(stage1.get("raw", "")))

        feedback = stage_data.get("stage2_feedback")
        if feedback:
            fb_dir = os.path.join(cand_dir, "stage2_feedback_review")
            os.makedirs(fb_dir, exist_ok=True)
            with open(os.path.join(fb_dir, "feedback.txt"), "w") as f:
                f.write(str(feedback.get("feedback", "")))
            with open(os.path.join(fb_dir, "repair_advice.txt"), "w") as f:
                f.write(str(feedback.get("repair_advice", "")))
            with open(os.path.join(fb_dir, "raw.txt"), "w") as f:
                f.write(str(feedback.get("raw", "")))

        stage2 = stage_data.get("stage2_nl")
        if stage2:
            stage_dir = os.path.join(cand_dir, "stage2_nl_revision")
            os.makedirs(stage_dir, exist_ok=True)
            with open(os.path.join(stage_dir, "summary.txt"), "w") as f:
                f.write(str(stage2.get("summary", "")))
            with open(os.path.join(stage_dir, "raw.txt"), "w") as f:
                f.write(str(stage2.get("raw", "")))

        return dom_path, prob_path

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        final_syn = final_sem = 0
        plan_not_found_total = plan_not_valid_total = 0
        rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
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
                    "final_syntactic_ok": syn_ok,
                    "final_semantic_ok": sem_ok,
                    "final_error_category": category,
                }
            )

        if rows:
            with open(os.path.join(out_dir, "metrics_nl_nl_pddl_per_problem.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        final_syn_acc = final_syn / len(problem_ids) if problem_ids else 0.0
        final_sem_acc = final_sem / len(problem_ids) if problem_ids else 0.0
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if final_sem == 0 else total_tokens / final_sem

        stage_rows = [
            {
                "stage": "stage1_nl_initial",
                "generated_count": len(problem_ids),
            },
            {
                "stage": "stage2_nl_revision",
                "generated_count": len(problem_ids),
            },
            {
                "stage": "stage3_final_pddl",
                "syntactic_success_count": final_syn,
                "semantic_success_count": final_sem,
                "syntactic_accuracy": final_syn_acc,
                "semantic_accuracy": final_sem_acc,
            },
        ]
        self._write_stage_metrics(out_dir, "metrics_nl_nl_pddl_stages.csv", stage_rows)

        with open(os.path.join(out_dir, "metrics_nl_nl_pddl_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
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
        }
        metrics.update(self._solver_error_metrics())
        return metrics

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "NLNLPDDLInference constructs prompts inside batch_generate_candidates."
        )
