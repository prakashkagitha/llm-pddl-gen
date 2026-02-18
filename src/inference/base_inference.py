"""
BaseInference – v4  (2025-05-15)
================================
Key additions
-------------
* get_prompt(pid)             – subclasses must supply (1 prompt / problem)
* get_batch_responses(ids)    – single-response per problem, batched
* get_multiple_batch_responses(ids, n) – pass@N helper, batched
* batch_generate_candidates(ids)       – default uses get_batch_responses
  Sub-classes override when multi-stage logic is needed.
* evaluate() now calls batch_generate_candidates once per experiment,
  not once per problem, guaranteeing that **every llm.generate** receives
  *all* prompts in one go.
"""
from __future__ import annotations

import os, re, gc, itertools, random, csv
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional

import torch
import numpy as np
from vllm import LLM, SamplingParams

from src.solver.dual_bfws_solver import DualBFWSSolver
from src.validation.validator import Validator


class BaseInference(ABC):
    CHAIN_OF_THOUGHT_SUFFIX = (
        "Let's think step by step, self-verify and explore different possibilities while "
        "following the given instruction."
    )

    _ERROR_CATEGORY_ORDER = (
        "duplicate_declarations",
        "type_or_arity_issues",
        "unsolvable",
        "miscellaneous",
    )
    _ERROR_CATEGORY_PATTERNS = {
        "duplicate_declarations": re.compile(
            r"(?i)(declared twice|multiply defined|already defined)"
        ),
        "type_or_arity_issues": re.compile(
            r"(?i)(type mismatch|unknown or empty type|unknown type|wrong type|"
            r"incompatible type|does not match type|declared to have \d+ \(not \d+\)|"
            r"number of arguments|expects \d+ arguments)"
        ),
        "unsolvable": re.compile(
            r"(?im)(plan not found|no plan|no solution|simplified to false|notfound|"
            r"search exhausted|search failed|^---\s*ok)"
        ),
    }


    def _augment_prompt(self, prompt: str) -> str:
        """Prepends the system prompt (if any) and adds the CoT suffix."""
        suffix = self.CHAIN_OF_THOUGHT_SUFFIX
        body = prompt.strip()

        sys_prompt = getattr(self, "system_prompt", "").strip()
        if sys_prompt:
            if not body:
                body = sys_prompt
            elif not body.startswith(sys_prompt):
                body = f"{sys_prompt}\n\n{body}"

        if not body:
            return suffix

        if not body.endswith(suffix):
            body = f"{body}\n\n{suffix}"

        return body
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        model: str,
        temperature: float,
        prompt_version: str,
        domain: str,
        data_type: str,
        *,
        tensor_parallel: int = 1,
        seed: int = 42,
        system_prompt: str | None = None,
        **_ignored,
    ):
        self.model = model
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.domain = domain
        self.data_type = data_type
        self.seed = seed
        self.system_prompt = (system_prompt or "").strip()

        # Set seeds for reproducibility
        self._set_seeds(seed)

        # Track prompt/completion token usage per evaluation run
        self._token_usage = {"prompt": 0, "completion": 0}
        self._stage_token_totals: Dict[str, int] = {}
        self._reset_error_counters()

        self.llm = LLM(
            model=model, 
            max_model_len=40_000, 
            tensor_parallel_size=tensor_parallel,
            seed=seed  # vLLM seed parameter
        )
        self.sampler = SamplingParams(
            temperature=temperature, 
            top_p=0.95, 
            max_tokens=20_000,
            seed=seed  # Sampling seed
        )

        self._solver = DualBFWSSolver()
        self._validator = Validator()

    def _set_seeds(self, seed: int):
        """Set seeds for all random number generators."""
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # # Make PyTorch deterministic (may impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        
        # Set environment variable for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)

    # ------------------------------------------------------------------ #
    #  Solver error categorisation                                       #
    # ------------------------------------------------------------------ #
    def _reset_error_counters(self) -> None:
        self._error_counts: Dict[str, int] = {
            key: 0 for key in self._ERROR_CATEGORY_ORDER
        }
        self._stage_error_counts: Dict[str, Dict[str, int]] = {}

    def _categorize_solver_error(self, msg: str) -> str:
        if not msg or not msg.strip():
            return ""
        for key in self._ERROR_CATEGORY_ORDER:
            pattern = self._ERROR_CATEGORY_PATTERNS.get(key)
            if pattern and pattern.search(msg):
                return key
        return "miscellaneous"

    def _record_solver_error(self, msg: str, stage: Optional[str] = None) -> str:
        category = self._categorize_solver_error(msg)
        if category:
            self._error_counts[category] += 1
            if stage:
                bucket = self._stage_error_counts.setdefault(
                    stage,
                    {key: 0 for key in self._ERROR_CATEGORY_ORDER},
                )
                bucket[category] += 1
        return category

    def _solver_error_metrics(self, stage: Optional[str] = None) -> Dict[str, int]:
        source: Dict[str, int]
        if stage is None:
            source = self._error_counts
        else:
            source = self._stage_error_counts.get(
                stage,
                {key: 0 for key in self._ERROR_CATEGORY_ORDER},
            )
        return {
            "duplicate_declaration_error_count": source["duplicate_declarations"],
            "type_or_arity_error_count": source["type_or_arity_issues"],
            "unsolvable_error_count": source["unsolvable"],
            "miscellaneous_error_count": source["miscellaneous"],
        }

    # ------------------------------------------------------------------ #
    #  Every pipeline must at least implement *get_prompt*               #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def get_prompt(self, problem_id: str) -> str: ...

    # ------------------------------------------------------------------ #
    #  Batched helpers – always use llm.generate on the full prompt list #
    # ------------------------------------------------------------------ #
    def get_batch_responses(self, problem_ids: List[str]) -> List[Dict]:
        """One answer per problem, single vLLM call."""
        prompts = [self._augment_prompt(self.get_prompt(pid)) for pid in problem_ids]
        outs = self._tracked_generate(prompts, self.sampler)
        return [self._resp2dict(o.outputs[0].text) for o in outs]

    def get_multiple_batch_responses(
        self, problem_ids: List[str], n: int
    ) -> List[List[Dict]]:
        """pass@N – *n* answers per problem, single vLLM call."""
        prm = [self._augment_prompt(self.get_prompt(pid)) for pid in problem_ids]
        sp = SamplingParams(
            temperature=self.temperature, 
            top_p=0.95, 
            max_tokens=20_000, 
            n=n,
            seed=self.seed  # Use instance seed
        )
        outs = self._tracked_generate(prm, sp)
        return [
            [self._resp2dict(o.text) for o in item.outputs] for item in outs
        ]

    # ------------------------------------------------------------------ #
    #  Optional override when a pipeline is multi-stage                  #
    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, problem_ids: List[str]) -> List[List[Dict]]:
        """Default = 1-shot pipelines → one candidate per problem."""
        return [[r] for r in self.get_batch_responses(problem_ids)]

    # ------------------------------------------------------------------ #
    #  Tiny utilities                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe(txt: str | None) -> str: return "" if txt is None else txt

    @staticmethod
    def _strip_think(txt: str) -> str:
        if not txt:
            return ""
        if "</think>" not in txt:
            return txt
        head, tail = txt.split("</think>", 1)
        if tail.strip():
            return tail
        return txt

    @staticmethod
    def _extract_last_tag(txt: str, tag: str) -> str:
        pattern = re.compile(fr"<{tag}>(.*?)</{tag}>", re.DOTALL)
        matches = list(pattern.finditer(txt))
        if not matches:
            return ""
        candidate = matches[-1].group(1)
        open_tag = f"<{tag}>"
        last_idx = candidate.rfind(open_tag)
        if last_idx != -1:
            candidate = candidate[last_idx + len(open_tag):]
        return candidate.strip()

    def _unwrap(self, txt: str, tag: str) -> str:
        stripped = self._strip_think(txt)
        chunks = []
        if stripped:
            chunks.append(stripped)
        if stripped != txt:
            chunks.append(txt)
        elif not chunks:
            chunks.append(txt)

        for chunk in chunks:
            candidate = self._extract_last_tag(chunk, tag)
            if candidate:
                return candidate
        return ""

    def _resp2dict(self, full: str, summary: str = "") -> Dict[str, str]:
        return {
            "df": self._safe(self._unwrap(full, "domain_file")),
            "pf": self._safe(self._unwrap(full, "problem_file")),
            "raw": self._safe(full),
            "summary": summary,
        }

    # ---------------- token accounting -------------------------------- #
    def _reset_token_usage(self) -> None:
        self._token_usage = {"prompt": 0, "completion": 0}
        self._stage_token_totals = {}
        self._reset_error_counters()

    def _tracked_generate(self, prompts: List[str], sampling_params: SamplingParams):
        outs = self.llm.generate(prompts, sampling_params)

        prompt_total = 0
        completion_total = 0

        for out in outs:
            prompt_ids = getattr(out, "prompt_token_ids", None)
            if prompt_ids is None:
                prompt_len = len(getattr(out, "prompt_token_ids", []) or [])
            else:
                prompt_len = len(prompt_ids)
            prompt_total += prompt_len

            outputs = getattr(out, "outputs", []) or []
            for resp in outputs:
                token_ids = getattr(resp, "token_ids", None)
                if token_ids is None:
                    completion_len = len(getattr(resp, "token_ids", []) or [])
                else:
                    completion_len = len(token_ids)
                completion_total += completion_len

        self._token_usage["prompt"] += prompt_total
        self._token_usage["completion"] += completion_total
        return outs

    def _tracked_generate_token_free(
        self, prompts: List[str], sampling_params: SamplingParams
    ):
        """Run generation without counting tokens towards aggregate metrics."""
        prev_prompt = self._token_usage["prompt"]
        prev_completion = self._token_usage["completion"]
        outs = self._tracked_generate(prompts, sampling_params)

        prompt_delta = self._token_usage["prompt"] - prev_prompt
        completion_delta = self._token_usage["completion"] - prev_completion

        if prompt_delta:
            self._token_usage["prompt"] -= prompt_delta
        if completion_delta:
            self._token_usage["completion"] -= completion_delta

        return outs

    def _total_token_count(self) -> int:
        return self._token_usage["prompt"] + self._token_usage["completion"]

    def _mark_stage_tokens(self, stage: str) -> None:
        """Record cumulative token usage once a stage has completed generation."""
        self._stage_token_totals[stage] = self._total_token_count()

    def _write_stage_metrics(
        self, out_root: str, filename: str, rows: List[Dict[str, object]]
    ) -> None:
        """Persist per-stage aggregate metrics for quick inspection."""
        if not rows:
            return

        all_fields: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    all_fields.append(key)

        path = os.path.join(out_root, filename)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            norm_rows = [{k: row.get(k, "") for k in all_fields} for row in rows]
            writer.writerows(norm_rows)

    # ---------------- artefact-saving (unchanged from v3) -------------- #
    def _save_candidate(
        self, out_root: str, pid: str, idx: int, cand: Dict[str, str]
    ) -> Tuple[str, str]:
        """
        Save all artefacts for a single candidate.

        • df  → domain.pddl
        • pf  → problem.pddl
        • raw → raw.txt
        • any other key  → <key>.txt
        """
        cdir = os.path.join(out_root, pid, f"cand_{idx:02}")
        os.makedirs(cdir, exist_ok=True)

        # always present
        dom_path = os.path.join(cdir, "domain.pddl")
        prob_path = os.path.join(cdir, "problem.pddl")
        with open(dom_path, "w") as f:
            f.write(cand.get("df", ""))
        with open(prob_path, "w") as f:
            f.write(cand.get("pf", ""))
        with open(os.path.join(cdir, "raw.txt"), "w") as f:
            f.write(cand.get("raw", ""))

        # everything else → <key>.txt
        for k, v in cand.items():
            if k in {"df", "pf", "raw"}:
                continue
            fname = f"{k}.txt"
            with open(os.path.join(cdir, fname), "w") as f:
                f.write(str(v))

        return dom_path, prob_path


    # ---------------- candidate-level evaluation (unchanged) ----------- #
    def _check_candidate(
        self,
        pid: str,
        idx: int,
        cand: Dict[str, str],
        out_root: str,
        *,
        stage_name: Optional[str] = None,
    ) -> Tuple[bool, bool, str, str, bool, str]:
        df_path, pf_path = self._save_candidate(out_root, pid, idx, cand)
        plan, sol_err = self._solver.solve_with_error(df_path, pf_path)
        category = self._record_solver_error(sol_err, stage=stage_name)
        # print("Plan:\n", plan)
        # print("\nsol_err:\n", sol_err)
        syn_ok = bool(plan) or ("syntax error" not in sol_err.lower())
        if plan:
            plan_path = os.path.join(out_root, pid, f"cand_{idx:02}", "plan.txt")
            with open(plan_path, "w") as f: f.write(str(plan))
            val_ok, val_msg = self._validator.validate_with_error(
                self.domain, pid, plan_path
            )
            return syn_ok, val_ok, sol_err, val_msg, True, category
        return syn_ok, False, sol_err, "", False, category

    # ------------------------------------------------------------------ #
    #  Metric computation (now batched)                                  #
    # ------------------------------------------------------------------ #
    def evaluate(
        self, problem_ids: List[str], out_dir: str
    ) -> Dict[str, Union[int, float]]:
        self._reset_token_usage()
        all_cands = self.batch_generate_candidates(problem_ids)

        syn_ok = sem_ok = 0
        plan_not_found = plan_not_valid = 0

        for pid, cand_list in zip(problem_ids, all_cands):
            s_ok = se_ok = False
            plan_found_for_problem = False
            final_category = ""

            for idx, cand in enumerate(cand_list):
                so, seo, _, _, plan_found, category = self._check_candidate(
                    pid, idx, cand, out_dir, stage_name="stage_final"
                )
                s_ok |= so
                se_ok |= seo
                plan_found_for_problem |= plan_found
                if category:
                    final_category = category
                if se_ok:
                    break

            syn_ok += int(s_ok)
            sem_ok += int(se_ok)
            if se_ok:
                continue
            if final_category == "unsolvable":
                plan_not_found += 1
            elif plan_found_for_problem:
                plan_not_valid += 1
            else:
                plan_not_found += 1

        n = len(problem_ids)
        total_tokens = self._total_token_count()
        tokens_per_valid = float("nan") if sem_ok == 0 else total_tokens / sem_ok

        metrics = {
            "syntactic_accuracy": syn_ok / n if n else 0.0,
            "semantic_accuracy": sem_ok / n if n else 0.0,
            "syntactic_success_count": syn_ok,
            "semantic_success_count": sem_ok,
            "plan_not_found_count": plan_not_found,
            "plan_not_valid_count": plan_not_valid,
            "total_tokens": total_tokens,
            "tokens_per_valid_plan": tokens_per_valid,
            "n_problems": n,
        }
        metrics.update(self._solver_error_metrics())
        return metrics

    # ------------------------------------------------------------------ #
    def close(self):
        try: del self.llm
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()
