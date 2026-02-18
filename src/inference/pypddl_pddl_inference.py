"""
PyPDDL → PDDL two-stage pipeline.

Stage 1 converts natural-language descriptions into PyPDDL domain and
problem classes.  The generated Python is checked for basic syntax and
run through ``py2pddl`` so we can evaluate whether it yields valid PDDL.

Stage 2 consumes the PyPDDL program alongside the original textual
specification to produce the final PDDL artefacts.  We persist the stage
1 outputs (source, parsed PDDL, solver traces) and report per-stage
metrics mirroring the other PyPDDL pipelines.
"""
from __future__ import annotations

import csv
import os
import tempfile
from typing import Dict, List, Tuple

from py2pddl import parse as py2pddl_parse

from .base_inference import BaseInference


class PyPDDLPDDLInference(BaseInference):
    """Two-pass pipeline: PyPDDL intermediate → final PDDL."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stage_cache: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pypddl_prefix() -> str:
        with open("prompts/pypddl_instruction.txt") as f:
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
    def _solve_pddl(self, pid: str, domain_txt: str, problem_txt: str) -> Dict[str, object]:
        info = {
            "plan": "",
            "solver_error": "",
            "validator_msg": "",
            "syntactic_ok": False,
            "semantic_ok": False,
        }
        if not domain_txt.strip() or not problem_txt.strip():
            info["solver_error"] = "Empty domain/problem provided."
            self._record_solver_error(info["solver_error"], stage="stage1_pypddl")
            return info

        with tempfile.TemporaryDirectory(prefix="pypddl_stage_") as tmp:
            dom_path = os.path.join(tmp, "domain.pddl")
            prob_path = os.path.join(tmp, "problem.pddl")
            with open(dom_path, "w") as f:
                f.write(domain_txt)
            with open(prob_path, "w") as f:
                f.write(problem_txt)

            plan, solver_err = self._solver.solve_with_error(dom_path, prob_path)
            info["solver_error"] = solver_err or ""
            info["plan"] = str(plan) if plan else ""
            info["syntactic_ok"] = bool(plan) or "syntax error" not in (solver_err or "").lower()

            if info["solver_error"]:
                self._record_solver_error(info["solver_error"], stage="stage1_pypddl")

            if plan:
                plan_path = os.path.join(tmp, "plan.txt")
                with open(plan_path, "w") as f:
                    f.write(str(plan))
                sem_ok, val_msg = self._validator.validate_with_error(self.domain, pid, plan_path)
                info["semantic_ok"] = sem_ok
                info["validator_msg"] = val_msg
        return info

    def _analyse_pypddl(self, pid: str, domain_py: str, problem_py: str) -> Dict[str, object]:
        combined = (domain_py or "").strip() + "\n\n" + (problem_py or "").strip()
        info: Dict[str, object] = {
            "domain_py": domain_py or "",
            "problem_py": problem_py or "",
            "python_exec_ok": True,
            "parse_ok": True,
            "parsed_domain_pddl": "",
            "parsed_problem_pddl": "",
            "syntactic_ok": False,
            "semantic_ok": False,
            "plan": "",
            "solver_error": "",
            "validator_msg": "",
        }
        try:
            compile(combined, "<string>", "exec")
        except Exception as exc:
            info["python_exec_ok"] = False
            info["solver_error"] = f"Python compile error: {exc}"
            self._record_solver_error(info["solver_error"], stage="stage1_pypddl")
            info["parse_ok"] = False
            return info

        with tempfile.TemporaryDirectory(prefix="pypddl_stage_py_") as tmp:
            src_path = os.path.join(tmp, "domain_problem.py")
            with open(src_path, "w") as f:
                f.write(combined)

            dom_out = os.path.join(tmp, "domain.pddl")
            prob_out = os.path.join(tmp, "problem.pddl")
            try:
                py2pddl_parse.parse(src_path, domain=dom_out, problem=prob_out)
            except Exception as exc:
                info["parse_ok"] = False
                info["solver_error"] = f"py2pddl error: {exc}"
                self._record_solver_error(info["solver_error"], stage="stage1_pypddl")
                return info

            dom_path = dom_out + ".pddl"
            prob_path = prob_out + ".pddl"
            if os.path.exists(dom_path):
                with open(dom_path) as f:
                    info["parsed_domain_pddl"] = f.read()
            if os.path.exists(prob_path):
                with open(prob_path) as f:
                    info["parsed_problem_pddl"] = f.read()

        if info["parsed_domain_pddl"] and info["parsed_problem_pddl"]:
            solve_info = self._solve_pddl(pid, info["parsed_domain_pddl"], info["parsed_problem_pddl"])
            info.update(
                {
                    "plan": solve_info.get("plan", ""),
                    "solver_error": solve_info.get("solver_error", ""),
                    "validator_msg": solve_info.get("validator_msg", ""),
                    "syntactic_ok": solve_info.get("syntactic_ok", False),
                    "semantic_ok": solve_info.get("semantic_ok", False),
                }
            )
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
                self._pypddl_prefix()
                + "### Domain description\n"
                + f"{dom}\n\n"
                + "### Problem description\n"
                + f"{prob}\n\n"
                + "Write the PyPDDL domain and problem classes. Wrap the domain in <domain_file>…</domain_file> "
                "and the problem in <problem_file>…</problem_file>."
            )
            stage1_prompts.append(self._augment_prompt(prompt))

        stage1_outs = self._tracked_generate(stage1_prompts, self.sampler)
        self._mark_stage_tokens("stage1_pypddl")

        stage1_infos: List[Dict[str, object]] = []
        for pid, out in zip(pids, stage1_outs):
            full = out.outputs[0].text
            domain_py = self._unwrap(full, "domain_file")
            problem_py = self._unwrap(full, "problem_file")
            info = self._analyse_pypddl(pid, domain_py, problem_py)
            info["raw"] = full.split("</think>", 1)[-1].strip()
            stage1_infos.append(info)

        stage2_prompts: List[str] = []
        pddl_prefix = self._pddl_prefix()
        for dom_desc, prob_desc, info in zip(domains, problems, stage1_infos):
            domain_py = str(info.get("domain_py", "")).strip()
            problem_py = str(info.get("problem_py", "")).strip()
            prompt = (
                pddl_prefix
                + "You are an expert PDDL engineer. Using the natural-language descriptions and the PyPDDL program below, "
                "produce syntactically correct, mutually consistent domain and problem PDDL files. Keep the original "
                "descriptions in mind throughout your answer.\n\n"
                "### Original domain description\n"
                f"{dom_desc}\n\n"
                "### Original problem description\n"
                f"{prob_desc}\n\n"
                "### PyPDDL domain class\n```python\n"
                f"{domain_py}\n```\n\n"
                "### PyPDDL problem class\n```python\n"
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
            stage_dir = os.path.join(cand_dir, "stage_pypddl")
            os.makedirs(stage_dir, exist_ok=True)

            with open(os.path.join(stage_dir, "domain.py"), "w") as f:
                f.write(str(stage.get("domain_py", "")))
            with open(os.path.join(stage_dir, "problem.py"), "w") as f:
                f.write(str(stage.get("problem_py", "")))
            with open(os.path.join(stage_dir, "raw.txt"), "w") as f:
                f.write(str(stage.get("raw", "")))
            with open(os.path.join(stage_dir, "metrics.txt"), "w") as f:
                f.write(
                    "python_exec_ok: {}\nparse_ok: {}\nsyntactic_ok: {}\nsemantic_ok: {}\n".format(
                        stage.get("python_exec_ok", False),
                        stage.get("parse_ok", False),
                        stage.get("syntactic_ok", False),
                        stage.get("semantic_ok", False),
                    )
                )

            if stage.get("parsed_domain_pddl"):
                with open(os.path.join(stage_dir, "domain_from_py2pddl.pddl"), "w") as f:
                    f.write(str(stage.get("parsed_domain_pddl", "")))
            if stage.get("parsed_problem_pddl"):
                with open(os.path.join(stage_dir, "problem_from_py2pddl.pddl"), "w") as f:
                    f.write(str(stage.get("parsed_problem_pddl", "")))
            if stage.get("plan"):
                with open(os.path.join(stage_dir, "plan.txt"), "w") as f:
                    f.write(str(stage.get("plan", "")))
            if stage.get("solver_error"):
                with open(os.path.join(stage_dir, "solver_error.txt"), "w") as f:
                    f.write(str(stage.get("solver_error", "")))
            if stage.get("validator_msg"):
                with open(os.path.join(stage_dir, "validator.log"), "w") as f:
                    f.write(str(stage.get("validator_msg", "")))

        return dom_path, prob_path

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        stage1_python = stage1_parse = stage1_syn = stage1_sem = 0
        stage1_plan_not_found = stage1_plan_not_valid = 0
        final_syn = final_sem = 0
        plan_not_found_total = plan_not_valid_total = 0
        rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
            stage = self._stage_cache.get(pid, {})
            stage1_python += int(bool(stage.get("python_exec_ok")))
            stage1_parse += int(bool(stage.get("parse_ok")))
            stage1_syn += int(bool(stage.get("syntactic_ok")))
            stage1_sem += int(bool(stage.get("semantic_ok")))
            stage1_category = self._categorize_solver_error(stage.get("solver_error", ""))
            stage1_plan_found = bool(stage.get("plan"))
            if not bool(stage.get("semantic_ok")):
                if stage1_category == "unsolvable":
                    stage1_plan_not_found += 1
                elif stage1_plan_found:
                    stage1_plan_not_valid += 1
                else:
                    stage1_plan_not_found += 1

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
                    "stage1_parse_ok": bool(stage.get("parse_ok")),
                    "stage1_syntactic_ok": bool(stage.get("syntactic_ok")),
                    "stage1_semantic_ok": bool(stage.get("semantic_ok")),
                    "stage1_error_category": stage1_category,
                    "final_syntactic_ok": syn_ok,
                    "final_semantic_ok": sem_ok,
                    "final_error_category": category,
                }
            )

        if rows:
            with open(os.path.join(out_dir, "metrics_pypddl_pddl_per_problem.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        n = len(problem_ids) or 1
        stage1_python_acc = stage1_python / n
        stage1_parse_acc = stage1_parse / n
        stage1_syn_acc = stage1_syn / n
        stage1_sem_acc = stage1_sem / n
        final_syn_acc = final_syn / n
        final_sem_acc = final_sem / n
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if final_sem == 0 else total_tokens / final_sem
        stage1_tokens_total = self._stage_token_totals.get("stage1_pypddl", total_tokens)
        stage2_tokens_total = self._stage_token_totals.get("stage2_final_pddl", total_tokens)
        stage1_tokens_per_valid = (
            float("nan") if stage1_sem == 0 else stage1_tokens_total / stage1_sem
        )

        stage1_errors = self._solver_error_metrics("stage1_pypddl")
        stage2_errors = self._solver_error_metrics("stage2_final_pddl")

        stage_rows = [
            {
                "stage": "stage1_pypddl",
                "python_success_count": stage1_python,
                "parse_success_count": stage1_parse,
                "syntactic_success_count": stage1_syn,
                "semantic_success_count": stage1_sem,
                "python_accuracy": stage1_python_acc,
                "parse_accuracy": stage1_parse_acc,
                "syntactic_accuracy": stage1_syn_acc,
                "semantic_accuracy": stage1_sem_acc,
                "stage1_python_accuracy": stage1_python_acc,
                "stage1_parse_accuracy": stage1_parse_acc,
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
                "stage": "stage2_final_pddl",
                "syntactic_success_count": final_syn,
                "semantic_success_count": final_sem,
                "syntactic_accuracy": final_syn_acc,
                "semantic_accuracy": final_sem_acc,
                "stage1_python_accuracy": stage1_python_acc,
                "stage1_parse_accuracy": stage1_parse_acc,
                "plan_not_found_count": plan_not_found_total,
                "plan_not_valid_count": plan_not_valid_total,
                "n_problems": len(problem_ids),
                "total_tokens": stage2_tokens_total,
                "tokens_per_valid_plan": tokens_per_valid,
                "stage1_syntactic_accuracy": stage1_syn_acc,
                "stage1_semantic_accuracy": stage1_sem_acc,
                "final_syntactic_accuracy": final_syn_acc,
                "final_semantic_accuracy": final_sem_acc,
                **stage2_errors,
            },
        ]
        self._write_stage_metrics(out_dir, "metrics_pypddl_pddl_stages.csv", stage_rows)

        with open(os.path.join(out_dir, "metrics_pypddl_pddl_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage1_python_accuracy",
                    "stage1_parse_accuracy",
                    "stage1_syntactic_accuracy",
                    "stage1_semantic_accuracy",
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
                    "stage1_parse_accuracy": stage1_parse_acc,
                    "stage1_syntactic_accuracy": stage1_syn_acc,
                    "stage1_semantic_accuracy": stage1_sem_acc,
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
            "stage1_parse_accuracy": stage1_parse_acc,
            "stage1_syntactic_accuracy": stage1_syn_acc,
            "stage1_semantic_accuracy": stage1_sem_acc,
        }
        metrics.update(self._solver_error_metrics())
        return metrics

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "PyPDDLPDDLInference constructs prompts inside batch_generate_candidates."
        )
