"""
JSON → PDDL two-stage pipeline.

Stage 1 converts natural-language descriptions into a JSON-PDDL
specification. The JSON is parsed, validated, deterministically
translated into PDDL, and solved to collect quality metrics.

Stage 2 consumes the JSON representation (plus source descriptions) and
asks the LLM to emit final PDDL. We evaluate both the deterministic
translation and the LLM output, mirroring the structure of the
PyPDDL → PDDL pipeline.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import Dict, List, Tuple

from src.utils.json_pddl_converter import convert_json_to_pddl

from .base_inference import BaseInference


class JSONPDDLInference(BaseInference):
    """Two-pass pipeline: JSON-PDDL intermediate → final PDDL."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stage_cache: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _json_prefix() -> str:
        with open("prompts/json_instruction.txt") as f:
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

        with tempfile.TemporaryDirectory(prefix="json_stage_") as tmp:
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

    def _validate_domain_schema(self, domain: object) -> List[str]:
        errors: List[str] = []
        if not isinstance(domain, dict):
            return ["domain object must be a JSON object."]

        required_keys = ["name", "requirements", "types", "predicates", "actions"]
        for key in required_keys:
            if key not in domain:
                errors.append(f"domain missing required key '{key}'.")

        if not isinstance(domain.get("requirements", []), list):
            errors.append("domain.requirements must be a list.")
        if not isinstance(domain.get("types", []), list):
            errors.append("domain.types must be a list.")
        if not isinstance(domain.get("predicates", []), list):
            errors.append("domain.predicates must be a list.")
        if not isinstance(domain.get("actions", []), list) or not domain.get("actions"):
            errors.append("domain.actions must be a non-empty list.")

        if "constants" in domain:
            errors.append("domain.constants are not supported; move symbols into problem.objects.")
        if "functions" in domain and not isinstance(domain["functions"], list):
            errors.append("domain.functions must be a list when present.")

        return errors

    def _validate_problem_schema(self, problem: object) -> List[str]:
        errors: List[str] = []
        if not isinstance(problem, dict):
            return ["problem object must be a JSON object."]

        required_keys = ["name", "domain", "objects", "init", "goal"]
        for key in required_keys:
            if key not in problem:
                errors.append(f"problem missing required key '{key}'.")

        if not isinstance(problem.get("objects", {}), dict):
            errors.append("problem.objects must be a dictionary keyed by type.")
        if not isinstance(problem.get("init", []), list):
            errors.append("problem.init must be a list.")
        if not isinstance(problem.get("goal", {}), dict):
            errors.append("problem.goal must be a JSON object.")
        if "constraints" in problem and not isinstance(problem["constraints"], list):
            errors.append("problem.constraints must be a list when present.")
        if "metric" in problem and not isinstance(problem["metric"], dict):
            errors.append("problem.metric must be an object when present.")

        return errors

    def _analyse_json(self, pid: str, spec_json: str) -> Dict[str, object]:
        info: Dict[str, object] = {
            "json_spec": spec_json or "",
            "json_formatted": spec_json.strip(),
            "domain_json": "",
            "problem_json": "",
            "json_parse_ok": False,
            "schema_ok": False,
            "schema_errors": "",
            "conversion_ok": False,
            "conversion_domain_pddl": "",
            "conversion_problem_pddl": "",
            "conversion_plan": "",
            "conversion_solver_error": "",
            "conversion_validator_msg": "",
            "conversion_syntactic_ok": False,
            "conversion_semantic_ok": False,
            "raw": "",
        }

        spec_json_stripped = (spec_json or "").strip()
        if not spec_json_stripped:
            info["schema_errors"] = "json_spec was empty."
            return info

        try:
            parsed = json.loads(spec_json_stripped)
        except json.JSONDecodeError as exc:
            info["schema_errors"] = f"JSON parse error: {exc}"
            return info

        info["json_parse_ok"] = True

        if isinstance(parsed, dict):
            domain = parsed.get("domain")
            problem = parsed.get("problem")

            domain_errors = self._validate_domain_schema(domain)
            problem_errors = self._validate_problem_schema(problem)
            errors = domain_errors + problem_errors

            if domain is not None:
                info["domain_json"] = json.dumps(domain, indent=2)
            if problem is not None:
                info["problem_json"] = json.dumps(problem, indent=2)

            info["json_formatted"] = json.dumps(parsed, indent=2)

            if not errors:
                info["schema_ok"] = True
                try:
                    dom_pddl, prob_pddl = convert_json_to_pddl(parsed)
                    info["conversion_ok"] = True
                    info["conversion_domain_pddl"] = dom_pddl
                    info["conversion_problem_pddl"] = prob_pddl
                    solve_info = self._solve_pddl(
                        pid,
                        dom_pddl,
                        prob_pddl,
                        stage_name="stage1_json",
                    )
                    info["conversion_plan"] = solve_info.get("plan", "")
                    info["conversion_solver_error"] = solve_info.get("solver_error", "")
                    info["conversion_validator_msg"] = solve_info.get("validator_msg", "")
                    info["conversion_syntactic_ok"] = bool(solve_info.get("syntactic_ok", False))
                    info["conversion_semantic_ok"] = bool(solve_info.get("semantic_ok", False))
                except Exception as exc:
                    info["conversion_solver_error"] = f"conversion error: {exc}"
                    self._record_solver_error(info["conversion_solver_error"], stage="stage1_json")
            else:
                info["schema_errors"] = "\n".join(errors)
        else:
            info["schema_errors"] = "top-level JSON must contain 'domain' and 'problem'."

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
                self._json_prefix()
                + "### Domain description\n"
                + f"{dom}\n\n"
                + "### Problem description\n"
                + f"{prob}\n\n"
                + "Produce the JSON-PDDL specification. Wrap the complete JSON document in <json_spec>…</json_spec>."
            )
            stage1_prompts.append(self._augment_prompt(prompt))

        stage1_outs = self._tracked_generate(stage1_prompts, self.sampler)
        self._mark_stage_tokens("stage1_json")

        stage1_infos: List[Dict[str, object]] = []
        for pid, out in zip(pids, stage1_outs):
            full = out.outputs[0].text
            spec_json = self._unwrap(full, "json_spec")
            if not spec_json.strip():
                spec_json = full.split("</think>", 1)[-1].strip()
            info = self._analyse_json(pid, spec_json)
            info["raw"] = full.split("</think>", 1)[-1].strip()
            stage1_infos.append(info)

        stage2_prompts: List[str] = []
        pddl_prefix = self._pddl_prefix()
        for dom_desc, prob_desc, info in zip(domains, problems, stage1_infos):
            json_spec = str(info.get("json_formatted", "")).strip()
            domain_json = str(info.get("domain_json", "")).strip()
            problem_json = str(info.get("problem_json", "")).strip()
            conv_domain = str(info.get("conversion_domain_pddl", "")).strip()
            conv_problem = str(info.get("conversion_problem_pddl", "")).strip()

            prompt_parts = [
                pddl_prefix,
                "You are an expert PDDL engineer. Using the natural-language descriptions and the JSON-PDDL specification below, "
                "produce syntactically correct and mutually consistent domain/problem PDDL files. Keep the JSON details aligned "
                "with the original descriptions throughout your answer.\n\n",
                "### Original domain description\n",
                f"{dom_desc}\n\n",
                "### Original problem description\n",
                f"{prob_desc}\n\n",
                "### JSON-PDDL specification\n```json\n",
                f"{json_spec}\n```\n\n",
            ]
            if domain_json and problem_json:
                prompt_parts.extend(
                    [
                        "### JSON domain object\n```json\n",
                        f"{domain_json}\n```\n\n",
                        "### JSON problem object\n```json\n",
                        f"{problem_json}\n```\n\n",
                    ]
                )
            if conv_domain and conv_problem:
                prompt_parts.extend(
                    [
                        "### Deterministic JSON→PDDL translation (domain)\n```pddl\n",
                        f"{conv_domain}\n```\n\n",
                        "### Deterministic JSON→PDDL translation (problem)\n```pddl\n",
                        f"{conv_problem}\n```\n\n",
                    ]
                )

            prompt_parts.append(
                "Wrap only the final domain PDDL in <domain_file>…</domain_file> and the final problem PDDL in <problem_file>…</problem_file>."
            )

            stage2_prompts.append(self._augment_prompt("".join(prompt_parts)))

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
    def _save_candidate(  # type: ignore[override]
        self, out_root: str, pid: str, idx: int, cand: Dict[str, str]
    ):
        stage = self._stage_cache.get(pid, {})
        dom_path, prob_path = super()._save_candidate(out_root, pid, idx, cand)

        if stage:
            cand_dir = os.path.join(out_root, pid, f"cand_{idx:02}")
            stage_dir = os.path.join(cand_dir, "stage_json")
            os.makedirs(stage_dir, exist_ok=True)

            with open(os.path.join(stage_dir, "json_spec.json"), "w") as f:
                f.write(str(stage.get("json_spec", "")))
            with open(os.path.join(stage_dir, "json_formatted.json"), "w") as f:
                f.write(str(stage.get("json_formatted", "")))
            if stage.get("domain_json"):
                with open(os.path.join(stage_dir, "domain.json"), "w") as f:
                    f.write(str(stage.get("domain_json", "")))
            if stage.get("problem_json"):
                with open(os.path.join(stage_dir, "problem.json"), "w") as f:
                    f.write(str(stage.get("problem_json", "")))
            if stage.get("conversion_domain_pddl"):
                with open(os.path.join(stage_dir, "domain_from_converter.pddl"), "w") as f:
                    f.write(str(stage.get("conversion_domain_pddl", "")))
            if stage.get("conversion_problem_pddl"):
                with open(os.path.join(stage_dir, "problem_from_converter.pddl"), "w") as f:
                    f.write(str(stage.get("conversion_problem_pddl", "")))
            if stage.get("conversion_plan"):
                with open(os.path.join(stage_dir, "plan.txt"), "w") as f:
                    f.write(str(stage.get("conversion_plan", "")))
            if stage.get("conversion_solver_error"):
                with open(os.path.join(stage_dir, "solver_error.txt"), "w") as f:
                    f.write(str(stage.get("conversion_solver_error", "")))
            if stage.get("conversion_validator_msg"):
                with open(os.path.join(stage_dir, "validator.log"), "w") as f:
                    f.write(str(stage.get("conversion_validator_msg", "")))
            if stage.get("schema_errors"):
                with open(os.path.join(stage_dir, "schema_errors.txt"), "w") as f:
                    f.write(str(stage.get("schema_errors", "")))
            if stage.get("raw"):
                with open(os.path.join(stage_dir, "raw.txt"), "w") as f:
                    f.write(str(stage.get("raw", "")))

            metrics_txt = (
                "json_parse_ok: {}\n"
                "schema_ok: {}\n"
                "conversion_ok: {}\n"
                "conversion_syntactic_ok: {}\n"
                "conversion_semantic_ok: {}\n"
            ).format(
                stage.get("json_parse_ok", False),
                stage.get("schema_ok", False),
                stage.get("conversion_ok", False),
                stage.get("conversion_syntactic_ok", False),
                stage.get("conversion_semantic_ok", False),
            )
            with open(os.path.join(stage_dir, "metrics.txt"), "w") as f:
                f.write(metrics_txt)

        return dom_path, prob_path

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        stage1_parse = stage1_schema = stage1_conversion = 0
        stage1_syn = stage1_sem = 0
        stage1_plan_not_found = stage1_plan_not_valid = 0
        final_syn = final_sem = 0
        plan_not_found_total = plan_not_valid_total = 0
        rows: List[Dict[str, object]] = []

        for pid, cand_list in zip(problem_ids, all_cands):
            cand = cand_list[0]
            stage = self._stage_cache.get(pid, {})
            stage1_parse += int(bool(stage.get("json_parse_ok")))
            stage1_schema += int(bool(stage.get("schema_ok")))
            stage1_conversion += int(bool(stage.get("conversion_ok")))
            stage1_syn += int(bool(stage.get("conversion_syntactic_ok")))
            stage1_sem += int(bool(stage.get("conversion_semantic_ok")))

            stage1_category = self._categorize_solver_error(stage.get("conversion_solver_error", ""))
            stage1_plan_found = bool(stage.get("conversion_plan"))
            if not bool(stage.get("conversion_semantic_ok")):
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
                    "stage1_json_parse_ok": bool(stage.get("json_parse_ok")),
                    "stage1_schema_ok": bool(stage.get("schema_ok")),
                    "stage1_conversion_ok": bool(stage.get("conversion_ok")),
                    "stage1_conversion_syntactic_ok": bool(stage.get("conversion_syntactic_ok")),
                    "stage1_conversion_semantic_ok": bool(stage.get("conversion_semantic_ok")),
                    "stage1_solver_error_category": stage1_category,
                    "stage1_solver_error": stage.get("conversion_solver_error", ""),
                    "final_syntactic_ok": syn_ok,
                    "final_semantic_ok": sem_ok,
                    "final_solver_error_category": category,
                }
            )

        if rows:
            with open(os.path.join(out_dir, "metrics_json_pddl_per_problem.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        n = len(problem_ids) or 1
        stage1_parse_acc = stage1_parse / n
        stage1_schema_acc = stage1_schema / n
        stage1_conversion_acc = stage1_conversion / n
        stage1_syn_acc = stage1_syn / n
        stage1_sem_acc = stage1_sem / n
        final_syn_acc = final_syn / n
        final_sem_acc = final_sem / n

        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if final_sem == 0 else total_tokens / final_sem
        stage1_tokens_total = self._stage_token_totals.get("stage1_json", total_tokens)
        stage2_tokens_total = self._stage_token_totals.get("stage2_final_pddl", total_tokens)
        stage1_tokens_per_valid = float("nan") if stage1_sem == 0 else stage1_tokens_total / stage1_sem

        stage1_errors = self._solver_error_metrics("stage1_json")
        stage2_errors = self._solver_error_metrics("stage2_final_pddl")

        stage_rows = [
            {
                "stage": "stage1_json",
                "json_parse_success_count": stage1_parse,
                "schema_success_count": stage1_schema,
                "conversion_success_count": stage1_conversion,
                "syntactic_success_count": stage1_syn,
                "semantic_success_count": stage1_sem,
                "json_parse_accuracy": stage1_parse_acc,
                "schema_accuracy": stage1_schema_acc,
                "conversion_accuracy": stage1_conversion_acc,
                "syntactic_accuracy": stage1_syn_acc,
                "semantic_accuracy": stage1_sem_acc,
                "plan_not_found_count": stage1_plan_not_found,
                "plan_not_valid_count": stage1_plan_not_valid,
                "n_problems": len(problem_ids),
                "total_tokens": stage1_tokens_total,
                "tokens_per_valid_plan": stage1_tokens_per_valid,
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
                "stage1_json_parse_accuracy": stage1_parse_acc,
                "stage1_schema_accuracy": stage1_schema_acc,
                "stage1_conversion_accuracy": stage1_conversion_acc,
                "stage1_syntactic_accuracy": stage1_syn_acc,
                "stage1_semantic_accuracy": stage1_sem_acc,
                "final_syntactic_accuracy": final_syn_acc,
                "final_semantic_accuracy": final_sem_acc,
                **stage2_errors,
            },
        ]
        self._write_stage_metrics(out_dir, "metrics_json_pddl_stages.csv", stage_rows)

        with open(os.path.join(out_dir, "metrics_json_pddl_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage1_json_parse_accuracy",
                    "stage1_schema_accuracy",
                    "stage1_conversion_accuracy",
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
                    "stage1_json_parse_accuracy": stage1_parse_acc,
                    "stage1_schema_accuracy": stage1_schema_acc,
                    "stage1_conversion_accuracy": stage1_conversion_acc,
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
            "stage1_json_parse_accuracy": stage1_parse_acc,
            "stage1_schema_accuracy": stage1_schema_acc,
            "stage1_conversion_accuracy": stage1_conversion_acc,
            "stage1_syntactic_accuracy": stage1_syn_acc,
            "stage1_semantic_accuracy": stage1_sem_acc,
        }
        metrics.update(self._solver_error_metrics())
        return metrics

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "JSONPDDLInference constructs prompts inside batch_generate_candidates."
        )
