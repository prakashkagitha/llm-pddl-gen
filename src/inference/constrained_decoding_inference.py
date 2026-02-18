"""
ConstrainedDecodingInference  (v4, 2025‑05‑17)
------------------------------------------------
CFG‑guided decoding with *outlines* (Transformers backend).

Key points
~~~~~~~~~~
* **No vLLM GPU engine** is created – avoids double‑initialisation and
  heavy GPU memory usage.  Only the HF‑Transformers model used by
  *outlines* is loaded.
* Inherits from `BaseInference` but overrides `__init__` so it *does not*
  call `super().__init__()`.  Instead it sets only the minimal fields
  that `BaseInference` needs for evaluation, while leaving `self.llm = None`.
* Batched generation: all domain prompts decoded in one call, then all
  problem prompts decoded in a second.
* Fully compatible with `BaseInference.evaluate()` via a custom
  `batch_generate_candidates` implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from outlines import models, samplers, generate

from .base_inference import BaseInference
from src.solver.dual_bfws_solver import DualBFWSSolver
from src.validation.validator import Validator


class ConstrainedDecodingInference(BaseInference):
    """CFG‑constrained decoding without vLLM."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        model: str,
        temperature: float,
        prompt_version: str,
        domain: str,
        data_type: str,
        *,
        grammar_path: str = "pddl3_1.lark",
        model_name: str | None = None,
        system_prompt: str | None = None,
        **_kw,  # absorb any extras (tensor_parallel, etc.)
    ):
        # ---- minimal BaseInference state (avoid vLLM) -----------------
        self.model = model
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.domain = domain
        self.data_type = data_type
        self.system_prompt = (system_prompt or "").strip()

        # No GPU‑heavy vLLM engine
        self.llm = None  # so any accidental call will fail fast

        # Shared downstream components
        self._solver = DualBFWSSolver()
        self._validator = Validator()

        # ------------ outlines‑specific initialisation -----------------
        self._cd_model = models.transformers(model_name or self.model, device="auto")
        self._sampler = samplers.multinomial(temperature=self.temperature, top_p=0.95)

        grammar_text = Path(grammar_path).read_text("utf-8")
        self._dom_gen = generate.cfg(
            self._cd_model,
            "?start: domain\n" + grammar_text,
            sampler=self._sampler,
        )
        self._prob_gen = generate.cfg(
            self._cd_model,
            "?start: problem\n" + grammar_text,
            sampler=self._sampler,
        )

        # cache instruction prefix (loaded lazily)
        self._instruction_prefix: str | None = None

    # ------------------------------------------------------------------ #
    # BaseInference requirements
    # ------------------------------------------------------------------ #
    def get_prompt(self, problem_id: str) -> str:  # noqa: D401, N802
        raise NotImplementedError(
            "ConstrainedDecodingInference constructs prompts in batches; "
            "this stub satisfies the ABC but is never used."
        )

    # ------------------------- helpers -------------------------------- #
    def _prefix(self) -> str:
        if self._instruction_prefix is None:
            with open("prompts/only_pddl_instruction.txt", encoding="utf-8") as f:
                self._instruction_prefix = f.read().rstrip() + "\n\n"
        return self._instruction_prefix

    def _descs(self, pid: str):
        base = Path("data") / f"textual_{self.domain}" / self.data_type
        with open(base / f"{pid}_domain.txt", encoding="utf-8") as f:
            dom = f.read()
        with open(base / f"{pid}_problem.txt", encoding="utf-8") as f:
            prob = f.read()
        return dom, prob

    # ------------------------------------------------------------------ #
    # Batched generation for every problem id.
    # ------------------------------------------------------------------ #
    def batch_generate_candidates(
        self, problem_ids: List[str]
    ) -> List[List[Dict[str, str]]]:
        # ---------- 1. DOMAIN FILES ------------------------------------ #
        dom_prompts: List[str] = []
        prob_descs: List[str] = []
        for pid in problem_ids:
            dom_desc, prob_desc = self._descs(pid)
            prob_descs.append(prob_desc)
            prompt = (
                self._prefix()
                + "DOMAIN DESCRIPTION:\n"
                + dom_desc
                + "\n\nDon't think, just Generate **only** the domain file in PDDL. /no_think \n"
            )
            dom_prompts.append(self._augment_prompt(prompt))
        # call _dom_gen 10 promts at a time
        domain_pddls: List[str] = []
        for i in range(0, len(dom_prompts), 10):
            dom_prompts_batch = dom_prompts[i : i + 10]
            domain_pddls.extend(self._dom_gen(dom_prompts_batch, max_tokens=2000))
        # domain_pddls = self._dom_gen(dom_prompts, max_tokens=2000)
        # add "(define (domain" back to the results
        # domain_pddls = ["(define (domain "+dom_pddl for dom_pddl in domain_pddls]

        # ---------- 2. PROBLEM FILES ----------------------------------- #
        prob_prompts: List[str] = []
        for dom_pddl, prob_desc in zip(domain_pddls, prob_descs):
            prompt = (
                self._prefix()
                + "DOMAIN FILE:\n"
                + dom_pddl
                + "\n\nPROBLEM DESCRIPTION:\n"
                + prob_desc
                + "\n\nDon't think, just generate **only** the problem file in PDDL. /no_think \n"
            )
            prob_prompts.append(self._augment_prompt(prompt))
        # call _prob_gen 10 promts at a time
        problem_pddls: List[str] = []
        for i in range(0, len(prob_prompts), 10):
            prob_prompts_batch = prob_prompts[i : i + 10]
            problem_pddls.extend(self._prob_gen(prob_prompts_batch, max_tokens=2000))
        # problem_pddls = self._prob_gen(prob_prompts, max_tokens=2000)
        # problem_pddls = ["(define (problem "+prob_pddl for prob_pddl in problem_pddls]

        # ---------- 3. Wrap outputs ------------------------------------ #
        out: List[List[Dict[str, str]]] = []
        for df_text, pf_text in zip(domain_pddls, problem_pddls):
            out.append([
                {
                    "df": df_text,
                    "pf": pf_text,
                    "raw": df_text + "\n" + pf_text,
                }
            ])
        return out

    # ------------------------------------------------------------------ #
    def close(self):
        try:
            del self._cd_model
        finally:
            super().close()
