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

import os, re, gc, itertools, random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
import numpy as np
from vllm import LLM, SamplingParams

from src.solver.dual_bfws_solver import DualBFWSSolver
from src.validation.validator import Validator


class BaseInference(ABC):
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
        llm=None,
        **_ignored,
    ):
        self.model = model
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.domain = domain
        self.data_type = data_type
        self.seed = seed

        # Set seeds for reproducibility
        self._set_seeds(seed)

        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLM(
                model=model, 
                max_model_len=30_000, 
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
    #  Every pipeline must at least implement *get_prompt*               #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def get_prompt(self, problem_id: str) -> str: ...

    # ------------------------------------------------------------------ #
    #  Batched helpers – always use llm.generate on the full prompt list #
    # ------------------------------------------------------------------ #
    def get_batch_responses(self, problem_ids: List[str]) -> List[Dict]:
        """One answer per problem, single vLLM call."""
        prompts = [self.get_prompt(pid) for pid in problem_ids]
        outs = self.llm.generate(prompts, self.sampler)
        return [self._resp2dict(o.outputs[0].text) for o in outs]

    def get_multiple_batch_responses(
        self, problem_ids: List[str], n: int
    ) -> List[List[Dict]]:
        """pass@N – *n* answers per problem, single vLLM call."""
        prm = [self.get_prompt(pid) for pid in problem_ids]
        sp = SamplingParams(
            temperature=self.temperature, 
            top_p=0.95, 
            max_tokens=20_000, 
            n=n,
            seed=self.seed  # Use instance seed
        )
        outs = self.llm.generate(prm, sp)
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
    def _unwrap(txt: str, tag: str) -> str:
        txt = txt.split("</think>", 1)[-1]
        m = re.search(fr"<{tag}>(.*?)</{tag}>", txt, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _resp2dict(self, full: str, summary: str = "") -> Dict[str, str]:
        full = full.split("</think>", 1)[-1]
        return {
            "df": self._safe(self._unwrap(full, "domain_file")),
            "pf": self._safe(self._unwrap(full, "problem_file")),
            "raw": full,
            "summary": summary,
        }

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
        self, pid: str, idx: int, cand: Dict[str, str], out_root: str
    ) -> Tuple[bool, bool, str, str]:
        df_path, pf_path = self._save_candidate(out_root, pid, idx, cand)
        plan, sol_err = self._solver.solve_with_error(df_path, pf_path)
        # print("Plan:\n", plan)
        # print("\nsol_err:\n", sol_err)
        syn_ok = bool(plan) or ("syntax error" not in sol_err.lower())
        if plan:
            plan_path = os.path.join(out_root, pid, f"cand_{idx:02}", "plan.txt")
            with open(plan_path, "w") as f: f.write(str(plan))
            val_ok, val_msg = self._validator.validate_with_error(
                self.domain, pid, plan_path
            )
            return syn_ok, val_ok, sol_err, val_msg
        return syn_ok, False, sol_err, ""

    # ------------------------------------------------------------------ #
    #  Metric computation (now batched)                                  #
    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str) -> Tuple[float, float]:
        all_cands = self.batch_generate_candidates(problem_ids)
        syn_ok = sem_ok = 0

        for pid, cand_list in zip(problem_ids, all_cands):
            s_ok = se_ok = False
            for idx, cand in enumerate(cand_list):
                so, seo, _, _ = self._check_candidate(pid, idx, cand, out_dir)
                s_ok |= so
                se_ok |= seo
                if se_ok: break
            syn_ok += int(s_ok)
            sem_ok += int(se_ok)

        n = len(problem_ids)
        return syn_ok / n, sem_ok / n

    # ------------------------------------------------------------------ #
    def close(self):
        try: del self.llm
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()
