"""
NL → PDDL two-stage pipeline.

Stage 1: produce a natural-language planning specification using
         prompts/nl_instruction.txt as the base instruction.
Stage 2: generate PDDL (domain + problem) using the NL summary plus
         the original textual descriptions.

Compared to summary_pddl_inference, this pipeline stores the NL summary
inside a dedicated stage_nl/ folder so future multi-stage pipelines can
keep artefacts per representation.
"""
from __future__ import annotations

import os
import shutil
from typing import Dict, List, Tuple

from .base_inference import BaseInference


class NLPDDLInference(BaseInference):
    """Two-pass pipeline: natural-language summary → PDDL."""

    # ------------------------------------------------------------------ #
    #  Prompt helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pddl_prefix() -> str:
        with open("prompts/only_pddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    @staticmethod
    def _nl_prefix() -> str:
        with open("prompts/nl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    def _descs(self, pid: str) -> Tuple[str, str]:
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f:
            dom = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f:
            prob = f.read()
        return dom, prob

    # ------------------------------------------------------------------ #
    #  Two-stage generation                                              #
    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        nl_prompts: List[str] = []
        domains: List[str] = []
        problems: List[str] = []

        for pid in pids:
            dom, prob = self._descs(pid)
            domains.append(dom)
            problems.append(prob)
            prompt = (
                self._nl_prefix()
                + (
                    "Use the following natural-language descriptions to craft a "
                    "complete planning specification in prose. Follow the layout "
                    "described above.\n\n"
                    "### Original domain description\n"
                    f"{dom}\n\n"
                    "### Original problem description\n"
                    f"{prob}\n\n"
                    "Wrap the final structured prose in <nl_summary>…</nl_summary>. "
                    "Begin your reasoning with <think> so we can inspect the chain of thought.\n"
                    "Do not emit any PDDL or code yet."
                )
            )
            nl_prompts.append(self._augment_prompt(prompt))

        nl_outs = self._tracked_generate(nl_prompts, self.sampler)

        summaries: List[str] = []
        nl_raw_texts: List[str] = []
        for out in nl_outs:
            full = out.outputs[0].text
            summary = self._unwrap(full, "nl_summary")
            if not summary:
                summary = full.split("</think>", 1)[-1].strip()
            summaries.append(summary)
            nl_raw_texts.append(full.split("</think>", 1)[-1].strip())

        pddl_prompts: List[str] = []
        pddl_prefix = self._pddl_prefix()
        for summary, dom, prob in zip(summaries, domains, problems):
            prompt = (
                pddl_prefix
                + (
                    "You are an expert PDDL engineer.\n"
                    "Leverage the structured natural-language planning "
                    "specification, along with the original descriptions, to "
                    "produce consistent domain and problem files. Always ground your "
                    "final answer in the original domain and problem descriptions shown "
                    "below.\n\n"
                    "### Original domain description\n"
                    f"{dom}\n\n"
                    "### Original problem description\n"
                    f"{prob}\n\n"
                    "### Natural-language planning specification\n"
                    f"{summary}\n\n"
                    "Output syntactically correct PDDL. Wrap only the domain file "
                    "in <domain_file>…</domain_file> and the problem file in "
                    "<problem_file>…</problem_file>."
                )
            )
            pddl_prompts.append(self._augment_prompt(prompt))

        pddl_outs = self._tracked_generate(pddl_prompts, self.sampler)

        results: List[List[Dict]] = []
        for out, summary, nl_raw in zip(pddl_outs, summaries, nl_raw_texts):
            cand = self._resp2dict(out.outputs[0].text, summary=summary)
            cand["nl_summary"] = summary
            cand["nl_raw"] = nl_raw
            results.append([cand])

        return results

    # ------------------------------------------------------------------ #
    #  Artefact saving: add stage-specific folder for NL outputs         #
    # ------------------------------------------------------------------ #
    def _save_candidate(self, out_root: str, pid: str, idx: int, cand: Dict[str, str]) -> Tuple[str, str]:
        dom_path, prob_path = super()._save_candidate(out_root, pid, idx, cand)

        cand_dir = os.path.join(out_root, pid, f"cand_{idx:02}")
        stage_nl_dir = os.path.join(cand_dir, "stage_nl")
        os.makedirs(stage_nl_dir, exist_ok=True)

        summary_txt = os.path.join(cand_dir, "summary.txt")
        if os.path.exists(summary_txt):
            shutil.move(summary_txt, os.path.join(stage_nl_dir, "summary.txt"))

        nl_raw_path = os.path.join(cand_dir, "nl_raw.txt")
        if os.path.exists(nl_raw_path):
            shutil.move(nl_raw_path, os.path.join(stage_nl_dir, "nl_raw.txt"))

        return dom_path, prob_path

    # ------------------------------------------------------------------ #
    def get_prompt(self, pid: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "NLPDDLInference constructs prompts inside batch_generate_candidates."
        )

    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        metrics = super().evaluate(problem_ids, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        stage_rows = [
            {
                "stage": "stage2_final_pddl",
                "syntactic_success_count": metrics.get("syntactic_success_count", 0),
                "semantic_success_count": metrics.get("semantic_success_count", 0),
                "syntactic_accuracy": metrics.get("syntactic_accuracy", 0.0),
                "semantic_accuracy": metrics.get("semantic_accuracy", 0.0),
            }
        ]
        self._write_stage_metrics(out_dir, "metrics_nl_pddl_stages.csv", stage_rows)
        return metrics
