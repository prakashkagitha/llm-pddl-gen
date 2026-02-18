"""
Python simulation → PDDL two-stage pipeline.

Stage 1 asks the LLM to synthesise a lightweight Python environment that
captures the planning dynamics (state representation, actions, helper
functions).  The generated code is checked for basic Python syntax and
archived so downstream tooling can inspect it.

Stage 2 converts that simulator – together with the original natural-
language descriptions – into final PDDL domain/problem files.  We store
stage-level artefacts and report metrics similar to the other pipelines.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

from .base_inference import BaseInference


class PythonSimPDDLInference(BaseInference):
    """Two-pass pipeline: Python simulator → PDDL."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stage_cache: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sim_prefix() -> str:
        with open("prompts/python_sim_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

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
    def _analyse_sim(self, domain_py: str, problem_py: str) -> Dict[str, object]:
        combined = (domain_py or "").strip() + "\n\n" + (problem_py or "").strip()
        info: Dict[str, object] = {
            "domain_py": domain_py or "",
            "problem_py": problem_py or "",
            "python_exec_ok": True,
            "compile_error": "",
        }
        try:
            compile(combined, "<string>", "exec")
        except Exception as exc:
            info["python_exec_ok"] = False
            info["compile_error"] = str(exc)
        return info

    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        stage1_prompts: List[str] = []
        domains: List[str] = []
        problems: List[str] = []
        for pid in pids:
            dom, prob = self._descs(pid)
            domains.append(dom)
            problems.append(prob)
            prompt = (
                self._sim_prefix()
                + "### Domain description\n"
                + f"{dom}\n\n"
                + "### Problem description\n"
                + f"{prob}\n\n"
                + "Synthesise the Python planning simulator. Wrap the reusable domain code in <domain_file>…</domain_file> "
                "and the problem-specific code in <problem_file>…</problem_file>."
            )
            stage1_prompts.append(self._augment_prompt(prompt))

        stage1_outs = self._tracked_generate(stage1_prompts, self.sampler)
        self._mark_stage_tokens("stage1_python_sim")

        stage1_infos: List[Dict[str, object]] = []
        for out in stage1_outs:
            full = out.outputs[0].text
            domain_py = self._unwrap(full, "domain_file")
            problem_py = self._unwrap(full, "problem_file")
            info = self._analyse_sim(domain_py, problem_py)
            info["raw"] = full.split("</think>", 1)[-1].strip()
            stage1_infos.append(info)

        stage2_prompts: List[str] = []
        pddl_prefix = self._pddl_prefix()
        for dom_desc, prob_desc, info in zip(domains, problems, stage1_infos):
            domain_py = str(info.get("domain_py", "")).strip()
            problem_py = str(info.get("problem_py", "")).strip()
            prompt = (
                pddl_prefix
                + "You are an expert PDDL engineer. Using the natural-language descriptions and the Python simulator below, "
                "produce syntactically correct and mutually consistent domain/problem PDDL files. Your answer must stay aligned "
                "with the original domain and problem descriptions referenced above.\n\n"
                "### Original domain description\n"
                f"{dom_desc}\n\n"
                "### Original problem description\n"
                f"{prob_desc}\n\n"
                "### Python simulator – domain module\n```python\n"
                f"{domain_py}\n```\n\n"
                "### Python simulator – problem module\n```python\n"
                f"{problem_py}\n```\n\n"
                "Wrap only the final domain PDDL in <domain_file>…</domain_file> and the final problem PDDL in <problem_file>…</problem_file>."
            )
            stage2_prompts.append(self._augment_prompt(prompt))

        final_outs = self._tracked_generate(stage2_prompts, self.sampler)
        self._mark_stage_tokens("stage2_final_pddl")

        results: List[List[Dict[str, str]]] = []
        self._stage_cache = {}
        for pid, out, info in zip(pids, final_outs, stage1_infos):
            cand = self._resp2dict(out.outputs[0].text)
            results.append([cand])
            self._stage_cache[pid] = info

        return results

    # ------------------------------------------------------------------ #
    def _save_candidate(
        self, out_root: str, pid: str, idx: int, cand: Dict[str, str]
    ):
        stage = self._stage_cache.get(pid, {})
        dom_path, prob_path = super()._save_candidate(out_root, pid, idx, cand)

        if stage:
            cand_dir = os.path.join(out_root, pid, f"cand_{idx:02}")
            stage_dir = os.path.join(cand_dir, "stage_pythonsim")
            os.makedirs(stage_dir, exist_ok=True)

            with open(os.path.join(stage_dir, "domain.py"), "w") as f:
                f.write(str(stage.get("domain_py", "")))
            with open(os.path.join(stage_dir, "problem.py"), "w") as f:
                f.write(str(stage.get("problem_py", "")))
            with open(os.path.join(stage_dir, "raw.txt"), "w") as f:
                f.write(str(stage.get("raw", "")))
            with open(os.path.join(stage_dir, "metrics.txt"), "w") as f:
                f.write("python_exec_ok: {}\n".format(stage.get("python_exec_ok", False)))
            if stage.get("compile_error"):
                with open(os.path.join(stage_dir, "compile_error.txt"), "w") as f:
                    f.write(str(stage.get("compile_error", "")))

        return dom_path, prob_path

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        stage1_python = 0
        final_syn = final_sem = 0
        plan_not_found_total = plan_not_valid_total = 0
        rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
            stage = self._stage_cache.get(pid, {})
            stage1_python += int(bool(stage.get("python_exec_ok")))

            syn_ok, sem_ok, sol_err, val_msg, plan_found, category = self._check_candidate(
                pid, 0, cand, out_dir, stage_name="stage2_final_pddl"
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
                    "stage1_python_exec_ok": bool(stage.get("python_exec_ok")),
                    "stage1_compile_error": stage.get("compile_error", ""),
                    "final_syntactic_ok": syn_ok,
                    "final_semantic_ok": sem_ok,
                    "solver_error_category": category,
                }
            )

        if rows:
            with open(os.path.join(out_dir, "metrics_pythonsim_pddl_per_problem.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        n = len(problem_ids) or 1
        stage1_python_acc = stage1_python / n
        final_syn_acc = final_syn / n
        final_sem_acc = final_sem / n
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if final_sem == 0 else total_tokens / final_sem
        stage1_tokens_total = self._stage_token_totals.get("stage1_python_sim", total_tokens)
        stage2_tokens_total = self._stage_token_totals.get("stage2_final_pddl", total_tokens)
        stage1_tokens_per_valid = float("nan")

        stage1_errors = self._solver_error_metrics("stage1_python_sim")
        stage2_errors = self._solver_error_metrics("stage2_final_pddl")

        stage_rows = [
            {
                "stage": "stage1_python_sim",
                "python_success_count": stage1_python,
                "python_accuracy": stage1_python_acc,
                "plan_not_found_count": 0,
                "plan_not_valid_count": 0,
                "n_problems": len(problem_ids),
                "total_tokens": stage1_tokens_total,
                "tokens_per_valid_plan": stage1_tokens_per_valid,
                "stage1_python_accuracy": stage1_python_acc,
                **stage1_errors,
            },
            {
                "stage": "stage2_final_pddl",
                "syntactic_success_count": final_syn,
                "semantic_success_count": final_sem,
                "syntactic_accuracy": final_syn_acc,
                "semantic_accuracy": final_sem_acc,
                "plan_not_found_count": plan_not_found_total,
                "plan_not_valid_count": plan_not_valid_total,
                "n_problems": len(problem_ids),
                "total_tokens": stage2_tokens_total,
                "tokens_per_valid_plan": tokens_per_valid,
                "stage1_python_accuracy": stage1_python_acc,
                "final_syntactic_accuracy": final_syn_acc,
                "final_semantic_accuracy": final_sem_acc,
                **stage2_errors,
            },
        ]
        self._write_stage_metrics(out_dir, "metrics_pythonsim_pddl_stages.csv", stage_rows)

        with open(os.path.join(out_dir, "metrics_pythonsim_pddl_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage1_python_accuracy",
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
                    "stage1_python_accuracy": stage1_python_acc,
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
            "stage1_python_accuracy": stage1_python_acc,
        }
        metrics.update(self._solver_error_metrics())
        return metrics

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "PythonSimPDDLInference constructs prompts inside batch_generate_candidates."
        )
