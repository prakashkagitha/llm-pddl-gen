"""
Plan generation pipeline.

Uses prompts/plan_instruction.txt to ask the LLM to act as a planner for
a given domain/problem pair and emit a validated plan.
Outputs per-problem diagnostics plus aggregate accuracies.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Dict, List

from .base_inference import BaseInference


class PlanGenInference(BaseInference):
    """Generate plans directly from natural-language descriptions."""

    @staticmethod
    def _prefix() -> str:
        with open("prompts/plan_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    def _descs(self, pid: str) -> Dict[str, str]:
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f:
            dom = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f:
            prob = f.read()
        return {"domain": dom, "problem": prob}

    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        prompts: List[str] = []
        for pid in pids:
            descs = self._descs(pid)
            prompt = (
                self._prefix()
                + "### Domain description\n"
                + f"{descs['domain']}\n\n"
                + "### Problem description\n"
                + f"{descs['problem']}\n"
                + "Generate the plan for this instance following the required format."
            )
            prompts.append(self._augment_prompt(prompt))

        outs = self._tracked_generate(prompts, self.sampler)

        results: List[List[Dict[str, str]]] = []
        self._last_candidates: Dict[str, Dict[str, str]] = {}

        for pid, out in zip(pids, outs):
            full = out.outputs[0].text
            plan = self._unwrap(full, "plan")
            plan = plan.strip()
            think_match = re.search(r"<think>(.*?)</think>", full, re.DOTALL)
            reasoning = think_match.group(1).strip() if think_match else ""
            raw_after_think = full.split("</think>", 1)[-1].strip()

            cand = {
                "plan": plan,
                "raw": raw_after_think,
                "reasoning": reasoning,
            }
            self._last_candidates[pid] = cand
            results.append([cand])

        return results

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        syn_ok_total = sem_ok_total = 0
        plan_not_found_total = plan_not_valid_total = 0
        per_problem_rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
            plan_text = cand.get("plan", "").strip()
            plan_lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
            syn_ok = bool(plan_lines)

            cand_dir = os.path.join(out_dir, pid, "cand_00")
            os.makedirs(cand_dir, exist_ok=True)

            reasoning = cand.get("reasoning", "")
            if reasoning:
                with open(os.path.join(cand_dir, "reasoning.txt"), "w") as f:
                    f.write(reasoning)

            raw = cand.get("raw", "")
            if raw:
                with open(os.path.join(cand_dir, "raw.txt"), "w") as f:
                    f.write(raw)

            plan_path = os.path.join(cand_dir, "plan.txt")
            with open(plan_path, "w") as f:
                plan_body = "\n".join(plan_lines)
                if plan_body:
                    f.write(plan_body)
                    if not plan_body.endswith("\n"):
                        f.write("\n")

            sem_ok = False
            val_msg = ""
            if syn_ok:
                sem_ok, val_msg = self._validator.validate_with_error(
                    self.domain, pid, plan_path
                )
            else:
                val_msg = "Plan empty."

            if val_msg:
                with open(os.path.join(cand_dir, "validator.log"), "w") as f:
                    f.write(val_msg)

            syn_ok_total += int(syn_ok)
            sem_ok_total += int(sem_ok)
            if not syn_ok:
                plan_not_found_total += 1
            elif not sem_ok:
                plan_not_valid_total += 1

            per_problem_rows.append(
                {
                    "problem": pid,
                    "plan_length": len(plan_lines),
                    "syntactic_ok": syn_ok,
                    "semantic_ok": sem_ok,
                    "validator_message": val_msg,
                }
            )

        if per_problem_rows:
            with open(os.path.join(out_dir, "metrics_plan_gen.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=per_problem_rows[0].keys())
                writer.writeheader()
                writer.writerows(per_problem_rows)

        n = len(problem_ids) or 1
        syn_acc = syn_ok_total / len(problem_ids) if problem_ids else 0.0
        sem_acc = sem_ok_total / len(problem_ids) if problem_ids else 0.0
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if sem_ok_total == 0 else total_tokens / sem_ok_total

        with open(os.path.join(out_dir, "metrics_plan_gen_summary.csv"), "w", newline="") as f:
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
                    "duplicate_declaration_error_count",
                    "type_or_arity_error_count",
                    "unsolvable_error_count",
                    "miscellaneous_error_count",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "syntactic_accuracy": syn_acc,
                    "semantic_accuracy": sem_acc,
                    "n_problems": len(problem_ids),
                    "plan_not_found_count": plan_not_found_total,
                    "plan_not_valid_count": plan_not_valid_total,
                    "total_tokens": total_tokens,
                    "tokens_per_valid_plan": tokens_per_valid,
                    **self._solver_error_metrics(),
                }
            )

        metrics = {
            "syntactic_accuracy": syn_acc,
            "semantic_accuracy": sem_acc,
            "syntactic_success_count": syn_ok_total,
            "semantic_success_count": sem_ok_total,
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
            "PlanGenInference constructs prompts inside batch_generate_candidates."
        )
