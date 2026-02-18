"""
PDDL → PyPDDL → PDDL pipeline.

Stage 1 generates PDDL straight from the natural-language descriptions.
Stage 2 converts the PDDL into PyPDDL, and Stage 3 produces a refined
PDDL pair from the PyPDDL program.  Artefacts from every stage are
persisted, and metrics cover Stage 1 syntactic/semantic success,
Stage 2 Python/parse rates, and final accuracies.
"""
from __future__ import annotations

import csv
import os
import tempfile
from typing import Dict, List, Tuple

from py2pddl import parse as py2pddl_parse

from .base_inference import BaseInference


class PDDLPyPDDLPDDLInference(BaseInference):
    """Three-stage pipeline: PDDL → PyPDDL → refined PDDL."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stage_cache: Dict[str, Dict[str, Dict[str, object]]] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pddl_prefix() -> str:
        with open("prompts/pddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    @staticmethod
    def _pypddl_prefix() -> str:
        with open("prompts/pypddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    def _descs(self, pid: str) -> Tuple[str, str]:
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f:
            dom = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f:
            prob = f.read()
        return dom, prob

    # ------------------------------------------------------------------ #
    def _solve_pddl(
        self,
        pid: str,
        domain_txt: str,
        problem_txt: str,
        *,
        stage_name: str,
    ) -> Dict[str, object]:
        info = {
            "plan": "",
            "solver_error": "",
            "validator_msg": "",
            "syntactic_ok": False,
            "semantic_ok": False,
        }
        if not domain_txt.strip() or not problem_txt.strip():
            info["solver_error"] = "Empty domain/problem provided."
            self._record_solver_error(info["solver_error"], stage=stage_name)
            return info

        with tempfile.TemporaryDirectory(prefix="pddl_stage1_") as tmp:
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
                self._record_solver_error(info["solver_error"], stage=stage_name)

            if plan:
                plan_path = os.path.join(tmp, "plan.txt")
                with open(plan_path, "w") as f:
                    f.write(str(plan))
                sem_ok, val_msg = self._validator.validate_with_error(self.domain, pid, plan_path)
                info["semantic_ok"] = sem_ok
                info["validator_msg"] = val_msg
        return info

    def _analyse_pypddl(self, pid: str, domain_py: str, problem_py: str) -> Dict[str, object]:
        combined = domain_py.strip() + "\n\n" + problem_py.strip()
        info: Dict[str, object] = {
            "domain_py": domain_py,
            "problem_py": problem_py,
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
            self._record_solver_error(info["solver_error"], stage="stage2_pypddl")
            info["parse_ok"] = False
            return info

        with tempfile.TemporaryDirectory(prefix="pddl_py_stage2_") as tmp:
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
                self._record_solver_error(info["solver_error"], stage="stage2_pypddl")
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
            solve_info = self._solve_pddl(
                pid,
                info["parsed_domain_pddl"],
                info["parsed_problem_pddl"],
                stage_name="stage2_pypddl",
            )
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
        dom_texts: List[str] = []
        prob_texts: List[str] = []
        stage1_prompts: List[str] = []
        for pid in pids:
            dom, prob = self._descs(pid)
            dom_texts.append(dom)
            prob_texts.append(prob)
            prompt = (
                self._pddl_prefix()
                + "### Domain description\n"
                + f"{dom}\n\n"
                + "### Problem description\n"
                + f"{prob}\n\n"
                + "Generate minimal PDDL domain and problem files. Wrap them inside "
                "<domain_file>…</domain_file> and <problem_file>…</problem_file>."
            )
            stage1_prompts.append(self._augment_prompt(prompt))

        stage1_outs = self._tracked_generate(stage1_prompts, self.sampler)
        self._mark_stage_tokens("stage1_initial_pddl")
        stage1_infos: List[Dict[str, object]] = []
        stage1_combined: List[str] = []
        for pid, out in zip(pids, stage1_outs):
            full = out.outputs[0].text
            domain_txt = self._unwrap(full, "domain_file")
            problem_txt = self._unwrap(full, "problem_file")
            info = self._solve_pddl(
                pid,
                domain_txt,
                problem_txt,
                stage_name="stage1_initial_pddl",
            )
            info.update({"domain": domain_txt, "problem": problem_txt, "raw": full.split("</think>", 1)[-1].strip()})
            stage1_infos.append(info)
            stage1_combined.append((domain_txt or "") + "\n\n" + (problem_txt or ""))

        stage2_prompts: List[str] = []
        for dom_desc, prob_desc, combined in zip(dom_texts, prob_texts, stage1_combined):
            prompt = (
                self._pypddl_prefix()
                + "Translate the PDDL specification below into PyPDDL domain and problem "
                "definitions.\n\n"
                "### Original domain description\n"
                f"{dom_desc}\n\n"
                "### Original problem description\n"
                f"{prob_desc}\n\n"
                "### PDDL specification\n"
                f"{combined}\n\n"
                "Wrap the domain code in <domain_file>…</domain_file> and the problem code "
                "in <problem_file>…</problem_file>."
            )
            stage2_prompts.append(self._augment_prompt(prompt))

        stage2_outs = self._tracked_generate(stage2_prompts, self.sampler)
        self._mark_stage_tokens("stage2_pypddl")
        stage2_infos: List[Dict[str, object]] = []
        stage2_combined: List[str] = []
        for pid, out in zip(pids, stage2_outs):
            full = out.outputs[0].text
            domain_py = self._unwrap(full, "domain_file")
            problem_py = self._unwrap(full, "problem_file")
            info = self._analyse_pypddl(pid, domain_py, problem_py)
            info["raw"] = full.split("</think>", 1)[-1].strip()
            stage2_infos.append(info)
            stage2_combined.append((domain_py or "") + "\n\n" + (problem_py or ""))

        stage3_prompts: List[str] = []
        for dom_desc, prob_desc, combined in zip(dom_texts, prob_texts, stage2_combined):
            prompt = (
                self._pddl_prefix()
                + "Refine the planning model represented in PyPDDL below and output "
                "consistent PDDL domain and problem files.\n\n"
                "### Original domain description\n"
                f"{dom_desc}\n\n"
                "### Original problem description\n"
                f"{prob_desc}\n\n"
                "### PyPDDL program\n"
                f"{combined}\n\n"
                "Wrap the domain in <domain_file>…</domain_file> and the problem in "
                "<problem_file>…</problem_file>."
            )
            stage3_prompts.append(self._augment_prompt(prompt))

        final_outs = self._tracked_generate(stage3_prompts, self.sampler)
        self._mark_stage_tokens("stage3_final_pddl")

        results: List[List[Dict[str, str]]] = []
        self._stage_cache = {}
        for pid, out, s1, s2 in zip(pids, final_outs, stage1_infos, stage2_infos):
            cand = self._resp2dict(out.outputs[0].text)
            results.append([cand])
            self._stage_cache[pid] = {
                "stage1": s1,
                "stage2": s2,
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
            stage1_dir = os.path.join(cand_dir, "stage1_pddl")
            os.makedirs(stage1_dir, exist_ok=True)
            with open(os.path.join(stage1_dir, "domain.pddl"), "w") as f:
                f.write(str(stage1.get("domain", "")))
            with open(os.path.join(stage1_dir, "problem.pddl"), "w") as f:
                f.write(str(stage1.get("problem", "")))
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
        if stage2:
            stage2_dir = os.path.join(cand_dir, "stage2_pypddl")
            os.makedirs(stage2_dir, exist_ok=True)
            with open(os.path.join(stage2_dir, "domain.py"), "w") as f:
                f.write(str(stage2.get("domain_py", "")))
            with open(os.path.join(stage2_dir, "problem.py"), "w") as f:
                f.write(str(stage2.get("problem_py", "")))
            with open(os.path.join(stage2_dir, "raw.txt"), "w") as f:
                f.write(str(stage2.get("raw", "")))
            with open(os.path.join(stage2_dir, "metrics.txt"), "w") as f:
                f.write(
                    "python_exec_ok: {}\nparse_ok: {}\nsyntactic_ok: {}\nsemantic_ok: {}\n".format(
                        stage2.get("python_exec_ok", False),
                        stage2.get("parse_ok", False),
                        stage2.get("syntactic_ok", False),
                        stage2.get("semantic_ok", False),
                    )
                )
            if stage2.get("parsed_domain_pddl"):
                with open(os.path.join(stage2_dir, "domain_from_py2pddl.pddl"), "w") as f:
                    f.write(str(stage2.get("parsed_domain_pddl", "")))
            if stage2.get("parsed_problem_pddl"):
                with open(os.path.join(stage2_dir, "problem_from_py2pddl.pddl"), "w") as f:
                    f.write(str(stage2.get("parsed_problem_pddl", "")))
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
        stage2_python = stage2_parse = 0
        stage2_syn = stage2_sem = 0
        stage1_plan_not_found = stage1_plan_not_valid = 0
        stage2_plan_not_found = stage2_plan_not_valid = 0
        final_syn = final_sem = 0
        plan_not_found_total = plan_not_valid_total = 0
        rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
            stages = self._stage_cache.get(pid, {})
            s1 = stages.get("stage1", {})
            s2 = stages.get("stage2", {})

            stage1_syn += int(bool(s1.get("syntactic_ok")))
            stage1_sem += int(bool(s1.get("semantic_ok")))
            stage1_category = self._categorize_solver_error(s1.get("solver_error", ""))
            stage1_plan_found = bool(s1.get("plan"))
            if not bool(s1.get("semantic_ok")):
                if stage1_category == "unsolvable":
                    stage1_plan_not_found += 1
                elif stage1_plan_found:
                    stage1_plan_not_valid += 1
                else:
                    stage1_plan_not_found += 1

            stage2_python += int(bool(s2.get("python_exec_ok")))
            stage2_parse += int(bool(s2.get("parse_ok")))
            stage2_syn += int(bool(s2.get("syntactic_ok")))
            stage2_sem += int(bool(s2.get("semantic_ok")))
            stage2_category = self._categorize_solver_error(s2.get("solver_error", ""))
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
                    "stage2_python_exec_ok": bool(s2.get("python_exec_ok")),
                    "stage2_parse_ok": bool(s2.get("parse_ok")),
                    "stage2_syntactic_ok": bool(s2.get("syntactic_ok")),
                    "stage2_semantic_ok": bool(s2.get("semantic_ok")),
                    "stage2_error_category": stage2_category,
                    "final_syntactic_ok": syn_ok,
                    "final_semantic_ok": sem_ok,
                    "final_error_category": category,
                }
            )

        if rows:
            with open(os.path.join(out_dir, "metrics_pddl_pypddl_pddl_per_problem.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        n = len(problem_ids) or 1
        stage1_syn_acc = stage1_syn / len(problem_ids) if problem_ids else 0.0
        stage1_sem_acc = stage1_sem / len(problem_ids) if problem_ids else 0.0
        stage2_python_acc = stage2_python / len(problem_ids) if problem_ids else 0.0
        stage2_parse_acc = stage2_parse / len(problem_ids) if problem_ids else 0.0
        stage2_syn_acc = stage2_syn / len(problem_ids) if problem_ids else 0.0
        stage2_sem_acc = stage2_sem / len(problem_ids) if problem_ids else 0.0
        final_syn_acc = final_syn / len(problem_ids) if problem_ids else 0.0
        final_sem_acc = final_sem / len(problem_ids) if problem_ids else 0.0
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if final_sem == 0 else total_tokens / final_sem
        stage1_tokens_total = self._stage_token_totals.get("stage1_initial_pddl", total_tokens)
        stage2_tokens_total = self._stage_token_totals.get("stage2_pypddl", total_tokens)
        stage3_tokens_total = self._stage_token_totals.get("stage3_final_pddl", total_tokens)
        stage1_tokens_per_valid = (
            float("nan") if stage1_sem == 0 else stage1_tokens_total / stage1_sem
        )
        stage2_tokens_per_valid = (
            float("nan") if stage2_sem == 0 else stage2_tokens_total / stage2_sem
        )

        stage1_errors = self._solver_error_metrics("stage1_initial_pddl")
        stage2_errors = self._solver_error_metrics("stage2_pypddl")
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
                "stage": "stage2_pypddl",
                "python_success_count": stage2_python,
                "parse_success_count": stage2_parse,
                "syntactic_success_count": stage2_syn,
                "semantic_success_count": stage2_sem,
                "python_accuracy": stage2_python_acc,
                "parse_accuracy": stage2_parse_acc,
                "syntactic_accuracy": stage2_syn_acc,
                "semantic_accuracy": stage2_sem_acc,
                "stage2_python_accuracy": stage2_python_acc,
                "stage2_parse_accuracy": stage2_parse_acc,
                "stage2_syntactic_accuracy": stage2_syn_acc,
                "stage2_semantic_accuracy": stage2_sem_acc,
                "plan_not_found_count": stage2_plan_not_found,
                "plan_not_valid_count": stage2_plan_not_valid,
                "n_problems": len(problem_ids),
                "total_tokens": stage2_tokens_total,
                "tokens_per_valid_plan": stage2_tokens_per_valid,
                "stage1_syntactic_accuracy": stage1_syn_acc,
                "stage1_semantic_accuracy": stage1_sem_acc,
                **stage2_errors,
            },
            {
                "stage": "stage3_final_pddl",
                "syntactic_success_count": final_syn,
                "semantic_success_count": final_sem,
                "syntactic_accuracy": final_syn_acc,
                "semantic_accuracy": final_sem_acc,
                "stage2_python_accuracy": stage2_python_acc,
                "stage2_parse_accuracy": stage2_parse_acc,
                "stage2_syntactic_accuracy": stage2_syn_acc,
                "stage2_semantic_accuracy": stage2_sem_acc,
                "plan_not_found_count": plan_not_found_total,
                "plan_not_valid_count": plan_not_valid_total,
                "n_problems": len(problem_ids),
                "total_tokens": stage3_tokens_total,
                "tokens_per_valid_plan": tokens_per_valid,
                "stage1_syntactic_accuracy": stage1_syn_acc,
                "stage1_semantic_accuracy": stage1_sem_acc,
                "final_syntactic_accuracy": final_syn_acc,
                "final_semantic_accuracy": final_sem_acc,
                **stage3_errors,
            },
        ]
        self._write_stage_metrics(out_dir, "metrics_pddl_pypddl_pddl_stages.csv", stage_rows)

        with open(os.path.join(out_dir, "metrics_pddl_pypddl_pddl_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage1_syntactic_accuracy",
                    "stage1_semantic_accuracy",
                    "stage2_python_accuracy",
                    "stage2_parse_accuracy",
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
                    "stage2_python_accuracy": stage2_python_acc,
                    "stage2_parse_accuracy": stage2_parse_acc,
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
            "stage2_python_accuracy": stage2_python_acc,
            "stage2_parse_accuracy": stage2_parse_acc,
            "stage2_syntactic_accuracy": stage2_syn_acc,
            "stage2_semantic_accuracy": stage2_sem_acc,
        }
        metrics.update(self._solver_error_metrics())
        return metrics

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "PDDLPyPDDLPDDLInference constructs prompts inside batch_generate_candidates."
        )
