"""
NL → PyPDDL → PDDL pipeline.

Stage 1 produces a natural-language planning specification using
prompts/nl_instruction.txt. Stage 2 converts that prose description into
PyPDDL code (domain + problem) via prompts/pypddl_instruction.txt. Stage
3 turns the PyPDDL program into a final pair of PDDL files using
prompts/pddl_instruction.txt.

Each intermediate artefact is stored under the candidate directory so
future pipelines can reuse them. Stage-level metrics track PyPDDL parsing
success and final syntactic / semantic accuracies.
"""
from __future__ import annotations

import csv
import os
import tempfile
from typing import Dict, List, Tuple

from py2pddl import parse as py2pddl_parse

from .base_inference import BaseInference


class NLPyPDDLPDDLInference(BaseInference):
    """Three-stage pipeline: NL summary → PyPDDL → PDDL."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._stage_cache: Dict[str, Dict[str, Dict[str, object]]] = {}

    # ------------------------------------------------------------------ #
    #  Prompt helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _nl_prefix() -> str:
        with open("prompts/nl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    @staticmethod
    def _pypddl_prefix() -> str:
        with open("prompts/pypddl_instruction.txt") as f:
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
    #  Small utilities                                                   #
    # ------------------------------------------------------------------ #
    def _solve_pddl(
        self,
        pid: str,
        domain_txt: str,
        problem_txt: str,
        *,
        stage_name: str,
    ) -> Dict[str, object]:
        """Run solver+validator on the provided PDDL strings."""
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

        with tempfile.TemporaryDirectory(prefix="nlpy_stage_") as tmp:
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
        """Compile PyPDDL, attempt py2pddl conversion, and (if possible) solve."""
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
        except Exception as exc:  # pragma: no cover - defensive
            info["python_exec_ok"] = False
            info["solver_error"] = f"Python compile error: {exc}"
            self._record_solver_error(info["solver_error"], stage="stage2_pypddl")
            info["parse_ok"] = False
            return info

        with tempfile.TemporaryDirectory(prefix="nlpy_pypddl_") as tmp:
            src_path = os.path.join(tmp, "domain_problem.py")
            with open(src_path, "w") as f:
                f.write(combined)

            domain_out = os.path.join(tmp, "domain.pddl")
            problem_out = os.path.join(tmp, "problem.pddl")
            try:
                py2pddl_parse.parse(src_path, domain=domain_out, problem=problem_out)
            except Exception as exc:  # pragma: no cover - defensive
                info["parse_ok"] = False
                info["solver_error"] = f"py2pddl error: {exc}"
                self._record_solver_error(info["solver_error"], stage="stage2_pypddl")
                return info

            domain_path = domain_out + ".pddl"
            problem_path = problem_out + ".pddl"
            if os.path.exists(domain_path):
                with open(domain_path) as f:
                    info["parsed_domain_pddl"] = f.read()
            if os.path.exists(problem_path):
                with open(problem_path) as f:
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

    @staticmethod
    def _format_solver_feedback(info: Dict[str, object]) -> str:
        parts: List[str] = []
        solver_err = str(info.get("solver_error", "")).strip()
        if solver_err:
            parts.append(f"Planner feedback:\n{solver_err}")

        validator_msg = str(info.get("validator_msg", "")).strip()
        if validator_msg:
            parts.append(f"Validator feedback:\n{validator_msg}")

        plan_txt = str(info.get("plan", "")).strip()
        if plan_txt:
            parts.append(f"Plan returned by solver:\n{plan_txt}")

        if info.get("semantic_ok"):
            parts.append(
                "The solver and validator accepted the plan. Protect correctness while addressing any stylistic issues."
            )
        elif not parts:
            parts.append(
                "The solver returned no diagnostics. Review the specification carefully to identify potential errors."
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        dom_texts: List[str] = []
        prob_texts: List[str] = []
        nl_prompts: List[str] = []
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
            nl_prompts.append(self._augment_prompt(prompt))

        nl_outs = self._tracked_generate(nl_prompts, self.sampler)
        stage1_summaries: List[str] = []
        stage1_raws: List[str] = []
        for out in nl_outs:
            full = out.outputs[0].text
            summary = self._unwrap(full, "nl_summary")
            if not summary:
                summary = full.split("</think>", 1)[-1].strip()
            stage1_summaries.append(summary)
            stage1_raws.append(full.split("</think>", 1)[-1].strip())

        pypddl_prompts: List[str] = []
        for summary, dom_desc, prob_desc in zip(stage1_summaries, dom_texts, prob_texts):
            prompt = (
                self._pypddl_prefix()
                + "Use the planning specification below to write PyPDDL domain and problem "
                "definitions.\n\n"
                "### Original domain description\n"
                f"{dom_desc}\n\n"
                "### Original problem description\n"
                f"{prob_desc}\n\n"
                "### Natural-language planning specification\n"
                f"{summary}\n\n"
                "Wrap the domain code inside <domain_file>…</domain_file> and the problem "
                "code inside <problem_file>…</problem_file>."
            )
            pypddl_prompts.append(self._augment_prompt(prompt))

        pypddl_outs = self._tracked_generate(pypddl_prompts, self.sampler)
        self._mark_stage_tokens("stage2_pypddl")
        stage2_infos: List[Dict[str, object]] = []
        stage2_combined: List[str] = []
        for pid, out in zip(pids, pypddl_outs):
            full = out.outputs[0].text
            domain_py = self._unwrap(full, "domain_file")
            problem_py = self._unwrap(full, "problem_file")
            info = self._analyse_pypddl(pid, domain_py, problem_py)
            info["raw"] = full.split("</think>", 1)[-1].strip()
            stage2_infos.append(info)
            stage2_combined.append((domain_py or "") + "\n\n" + (problem_py or ""))

        pddl_prompts: List[str] = []
        for dom_desc, prob_desc, combined, info in zip(dom_texts, prob_texts, stage2_combined, stage2_infos):
            parsed_domain = str(info.get("parsed_domain_pddl", "")).strip()
            parsed_problem = str(info.get("parsed_problem_pddl", "")).strip()
            feedback = self._format_solver_feedback(info)

            prompt_parts = [
                self._pddl_prefix(),
                "Use the PyPDDL program and solver feedback below to produce consistent PDDL domain and problem files.\n\n",
                "### Original domain description\n",
                f"{dom_desc}\n\n",
                "### Original problem description\n",
                f"{prob_desc}\n\n",
                "### PyPDDL program\n",
                f"{combined}\n\n",
            ]

            if parsed_domain or parsed_problem:
                prompt_parts.extend(
                    [
                        "### Auto-converted domain.pddl\n```pddl\n",
                        f"{parsed_domain}\n```\n\n",
                        "### Auto-converted problem.pddl\n```pddl\n",
                        f"{parsed_problem}\n```\n\n",
                    ]
                )

            prompt_parts.extend(
                [
                    "### Solver feedback\n",
                    f"{feedback}\n\n",
                    "Wrap the revised domain in <domain_file>…</domain_file> and the revised problem in <problem_file>…</problem_file>.",
                ]
            )

            prompt = "".join(prompt_parts)
            pddl_prompts.append(self._augment_prompt(prompt))

        final_outs = self._tracked_generate(pddl_prompts, self.sampler)
        self._mark_stage_tokens("stage3_final_pddl")

        results: List[List[Dict[str, str]]] = []
        self._stage_cache = {}
        for pid, out, summary, raw_nl, stage2 in zip(
            pids, final_outs, stage1_summaries, stage1_raws, stage2_infos
        ):
            cand = self._resp2dict(out.outputs[0].text, summary=summary)
            cand["reasoning"] = out.outputs[0].text.split("</think>", 1)[0] + "</think>"
            results.append([cand])
            self._stage_cache[pid] = {
                "stage1": {"summary": summary, "raw": raw_nl},
                "stage2": stage2,
            }
        return results

    # ------------------------------------------------------------------ #
    def _save_candidate(
        self, out_root: str, pid: str, idx: int, cand: Dict[str, str]
    ) -> Tuple[str, str]:
        stage_data = self._stage_cache.get(pid, {})
        filtered = {k: v for k, v in cand.items() if k not in {"reasoning"}}
        dom_path, prob_path = super()._save_candidate(out_root, pid, idx, filtered)

        cand_dir = os.path.join(out_root, pid, f"cand_{idx:02}")

        stage1 = stage_data.get("stage1")
        if stage1:
            nl_dir = os.path.join(cand_dir, "stage_nl")
            os.makedirs(nl_dir, exist_ok=True)
            with open(os.path.join(nl_dir, "summary.txt"), "w") as f:
                f.write(str(stage1.get("summary", "")))
            with open(os.path.join(nl_dir, "raw.txt"), "w") as f:
                f.write(str(stage1.get("raw", "")))

        stage2 = stage_data.get("stage2")
        if stage2:
            py_dir = os.path.join(cand_dir, "stage_pypddl")
            os.makedirs(py_dir, exist_ok=True)
            with open(os.path.join(py_dir, "domain.py"), "w") as f:
                f.write(str(stage2.get("domain_py", "")))
            with open(os.path.join(py_dir, "problem.py"), "w") as f:
                f.write(str(stage2.get("problem_py", "")))
            with open(os.path.join(py_dir, "raw.txt"), "w") as f:
                f.write(str(stage2.get("raw", "")))
            with open(os.path.join(py_dir, "metrics.txt"), "w") as f:
                f.write(
                    "python_exec_ok: {}\nparse_ok: {}\nsyntactic_ok: {}\nsemantic_ok: {}\n".format(
                        stage2.get("python_exec_ok", False),
                        stage2.get("parse_ok", False),
                        stage2.get("syntactic_ok", False),
                        stage2.get("semantic_ok", False),
                    )
                )
            if stage2.get("parsed_domain_pddl"):
                with open(os.path.join(py_dir, "domain_from_py2pddl.pddl"), "w") as f:
                    f.write(str(stage2.get("parsed_domain_pddl", "")))
            if stage2.get("parsed_problem_pddl"):
                with open(os.path.join(py_dir, "problem_from_py2pddl.pddl"), "w") as f:
                    f.write(str(stage2.get("parsed_problem_pddl", "")))
            if stage2.get("plan"):
                with open(os.path.join(py_dir, "plan.txt"), "w") as f:
                    f.write(str(stage2.get("plan", "")))
            if stage2.get("solver_error"):
                with open(os.path.join(py_dir, "solver_error.txt"), "w") as f:
                    f.write(str(stage2.get("solver_error", "")))
            if stage2.get("validator_msg"):
                with open(os.path.join(py_dir, "validator.log"), "w") as f:
                    f.write(str(stage2.get("validator_msg", "")))

        return dom_path, prob_path

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        parse_hits = python_hits = 0
        stage2_syn_total = stage2_sem_total = 0
        syn_total = sem_total = 0
        stage2_plan_not_found = stage2_plan_not_valid = 0
        plan_not_found_total = plan_not_valid_total = 0
        rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
            stage2 = self._stage_cache.get(pid, {}).get("stage2", {})
            python_ok = bool(stage2.get("python_exec_ok"))
            parse_ok = bool(stage2.get("parse_ok"))
            python_hits += int(python_ok)
            parse_hits += int(parse_ok)

            stage2_syn_total += int(bool(stage2.get("syntactic_ok")))
            stage2_sem_total += int(bool(stage2.get("semantic_ok")))

            syn_ok, sem_ok, sol_err, val_msg, plan_found, category = self._check_candidate(
                pid, 0, cand, out_dir, stage_name="stage3_final_pddl"
            )
            syn_total += int(syn_ok)
            sem_total += int(sem_ok)
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

            stage2_category = self._categorize_solver_error(stage2.get("solver_error", ""))
            if not bool(stage2.get("semantic_ok")):
                if stage2_category == "unsolvable":
                    stage2_plan_not_found += 1
                elif bool(stage2.get("plan")):
                    stage2_plan_not_valid += 1
                else:
                    stage2_plan_not_found += 1

            rows.append(
                {
                    "problem": pid,
                    "stage2_python_exec_ok": python_ok,
                    "stage2_parse_ok": parse_ok,
                    "stage2_syntactic_ok": bool(stage2.get("syntactic_ok")),
                    "stage2_semantic_ok": bool(stage2.get("semantic_ok")),
                    "stage2_error_category": stage2_category,
                    "final_syntactic_ok": syn_ok,
                    "final_semantic_ok": sem_ok,
                    "final_error_category": category,
                }
            )

        if rows:
            with open(os.path.join(out_dir, "metrics_nl_pypddl_pddl_per_problem.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        n = len(problem_ids) or 1
        python_acc = python_hits / len(problem_ids) if problem_ids else 0.0
        parse_acc = parse_hits / len(problem_ids) if problem_ids else 0.0
        stage2_syn_acc = stage2_syn_total / len(problem_ids) if problem_ids else 0.0
        stage2_sem_acc = stage2_sem_total / len(problem_ids) if problem_ids else 0.0
        syn_acc = syn_total / len(problem_ids) if problem_ids else 0.0
        sem_acc = sem_total / len(problem_ids) if problem_ids else 0.0
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if sem_total == 0 else total_tokens / sem_total
        stage2_tokens_total = self._stage_token_totals.get("stage2_pypddl", total_tokens)
        stage3_tokens_total = self._stage_token_totals.get("stage3_final_pddl", total_tokens)
        stage2_tokens_per_valid = (
            float("nan") if stage2_sem_total == 0 else stage2_tokens_total / stage2_sem_total
        )

        stage2_errors = self._solver_error_metrics("stage2_pypddl")
        stage3_errors = self._solver_error_metrics("stage3_final_pddl")

        stage_rows = [
            {
                "stage": "stage2_pypddl",
                "python_success_count": python_hits,
                "parse_success_count": parse_hits,
                "syntactic_success_count": stage2_syn_total,
                "semantic_success_count": stage2_sem_total,
                "python_accuracy": python_acc,
                "parse_accuracy": parse_acc,
                "syntactic_accuracy": stage2_syn_acc,
                "semantic_accuracy": stage2_sem_acc,
                "stage2_python_accuracy": python_acc,
                "stage2_parse_accuracy": parse_acc,
                "stage2_syntactic_accuracy": stage2_syn_acc,
                "stage2_semantic_accuracy": stage2_sem_acc,
                "plan_not_found_count": stage2_plan_not_found,
                "plan_not_valid_count": stage2_plan_not_valid,
                "n_problems": len(problem_ids),
                "total_tokens": stage2_tokens_total,
                "tokens_per_valid_plan": stage2_tokens_per_valid,
                **stage2_errors,
            },
            {
                "stage": "stage3_final_pddl",
                "syntactic_success_count": syn_total,
                "semantic_success_count": sem_total,
                "syntactic_accuracy": syn_acc,
                "semantic_accuracy": sem_acc,
                "plan_not_found_count": plan_not_found_total,
                "plan_not_valid_count": plan_not_valid_total,
                "n_problems": len(problem_ids),
                "total_tokens": stage3_tokens_total,
                "tokens_per_valid_plan": tokens_per_valid,
                "stage2_python_accuracy": python_acc,
                "stage2_parse_accuracy": parse_acc,
                "stage2_syntactic_accuracy": stage2_syn_acc,
                "stage2_semantic_accuracy": stage2_sem_acc,
                "final_syntactic_accuracy": syn_acc,
                "final_semantic_accuracy": sem_acc,
                **stage3_errors,
            },
        ]
        self._write_stage_metrics(out_dir, "metrics_nl_pypddl_pddl_stages.csv", stage_rows)

        with open(os.path.join(out_dir, "metrics_nl_pypddl_pddl_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage2_python_accuracy",
                    "stage2_parse_accuracy",
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
                    "stage2_python_accuracy": python_acc,
                    "stage2_parse_accuracy": parse_acc,
                    "final_syntactic_accuracy": syn_acc,
                    "final_semantic_accuracy": sem_acc,
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
            "syntactic_success_count": syn_total,
            "semantic_success_count": sem_total,
            "plan_not_found_count": plan_not_found_total,
            "plan_not_valid_count": plan_not_valid_total,
            "total_tokens": total_tokens,
            "tokens_per_valid_plan": tokens_per_valid,
            "n_problems": len(problem_ids),
            "stage2_python_accuracy": python_acc,
            "stage2_parse_accuracy": parse_acc,
        }
        metrics.update(self._solver_error_metrics())
        return metrics

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "NLPyPDDLPDDLInference builds prompts inside batch_generate_candidates."
        )
