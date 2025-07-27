from __future__ import annotations
import os, re, tempfile
from typing import List, Dict, Tuple

import pandas as pd
from vllm import SamplingParams

from .pddl_knowledge_inference import PDDLKnowledgeInference
from .summary_pddl_inference import SummaryPDDLInference

# --------------------------------------------------------------------------- #
#  Regular expressions for extracting VAL feedback                            #
# --------------------------------------------------------------------------- #
_VAL_ADVICE_RE = re.compile(r"Plan Repair Advice:(.*?)(?:Failed plans|$)", re.I | re.S)
_VAL_PLAN_RE   = re.compile(r"Plan to validate:(.*?)(?:Plan Validation details|$)", re.I | re.S)

# --------------------------------------------------------------------------- #
class SummaryRevisionSolverValInference(PDDLKnowledgeInference):
    """Summary PDDL + Solver **and** VAL-guided revision pipeline."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *a,
        k: int = 16,
        max_rounds: int = 4,
        llm=None,
        **kw,
    ):
        super().__init__(*a, **kw, llm=llm)
        self.k = k
        self.r = max_rounds
        # scratch dir – artefacts here are only for generating feedback
        self._tmp = tempfile.mkdtemp(prefix="revsolverval_")
        self.summary_pddl = SummaryPDDLInference(*a, **kw, llm=llm)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #
    def _extract_val_feedback(self, val_msg: str | None) -> str:
        """Return a (possibly empty) human‑readable VAL feedback string."""
        if not val_msg:
            return ""
        advice = _VAL_ADVICE_RE.search(val_msg)
        plan   = _VAL_PLAN_RE.search(val_msg)
        parts: List[str] = []
        if advice:
            parts.append(advice.group(1).strip())
        if plan:
            parts.append("Plan fragment under validation:\n" + plan.group(1).strip())
        # fallback – if regex failed just return the whole message (trimmed)
        return "\n\n".join(parts) if parts else val_msg.strip()

    def _build_revision_prompt(
        self,
        pid: str,
        cand: Dict[str, str],
        solver_fb: str,
        val_fb: str,
    ) -> str:
        """Construct the revision prompt combining *solver_fb* & *val_fb*."""
        sections: List[str] = [
            self.get_prompt(pid),
            "------ Previous attempt start ------",
            "[DOMAIN FILE]",
            cand["df"],
            "",
            "[PROBLEM FILE]",
            cand["pf"],
            "------ Previous attempt end --------",
            "",
        ]
        if solver_fb:
            sections.append("Feedback from the planning solver:\n" + solver_fb.strip() + "\n")
        if val_fb:
            sections.append("Feedback from the VAL validator:\n" + val_fb.strip() + "\n")

        sections.append(
            "Think step-by-step about what is wrong. Produce a **corrected** answer. "
            "Wrap the new domain file in <domain_file>…</domain_file> and the new problem file in "
            "<problem_file>…</problem_file>.\n<think>"
        )
        return "\n".join(sections)

    # ------------------------------------------------------------------ #
    #  Generation & revision loop                                        #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  Generation-and-revision loop – last-round candidates only         #
    # ------------------------------------------------------------------ #
    def batch_generate_candidates(
        self,
        pids: List[str],
    ) -> List[List[Dict]]:
        """
        Round-0: pass@k  →  each candidate gets tag  _round = 0

        For round r ≥ 1:
            • pick candidates with _round == r-1
            • if still semantically wrong → build revision prompt
            • the new answer is tagged _round = r
        """
        # ---- round-0 --------------------------------------------------
        attempts: List[List[Dict]] = self.summary_pddl.batch_generate_candidates(pids)
        for cand_list in attempts:
            for cand in cand_list:
                cand["_round"] = 0                       # tag origin

        # ---- rounds 1 … R --------------------------------------------
        for cur_r in range(1, self.r + 1):
            rev_prompts: List[str] = []
            tgt_map: List[int] = []                     # index into *attempts*

            # --- gather feedback & build prompts (only last-round cands)
            for p_idx, (pid, cand_list) in enumerate(zip(pids, attempts)):
                for c_idx, cand in enumerate(cand_list):
                    if cand["_round"] != cur_r - 1:
                        continue                        # not a last-round cand

                    # evaluate candidate  (feedback only – artefacts go to tmp)
                    _, sem_ok, sol_err, val_msg = self._check_candidate(
                        pid, c_idx, cand, self._tmp
                    )
                    if sem_ok:
                        continue                        # solved → skip

                    solver_fb = sol_err.strip() if sol_err else ""
                    val_fb    = self._extract_val_feedback(val_msg)

                    rev_prompts.append(
                        self._build_revision_prompt(pid, cand, solver_fb, val_fb)
                    )
                    tgt_map.append(p_idx)              # append to this problem

            # --- early exit ------------------------------------------
            if not rev_prompts:
                break

            # --- batched inference for all revision prompts ----------
            new_outs = self.llm.generate(rev_prompts, self.sampler)
            for p_idx, out in zip(tgt_map, new_outs):
                new_cand = self._resp2dict(out.outputs[0].text)
                new_cand["_round"] = cur_r            # tag creation round
                attempts[p_idx].append(new_cand)

        return attempts

    # ------------------------------------------------------------------ #
    #  Metric computation & artefact saving (same as RevSolver)          #
    # ------------------------------------------------------------------ #
    def evaluate(self, problem_ids: List[str], out_dir: str):  # type: ignore[override]
        # === generate / revise ======================================== #
        all_cands = self.batch_generate_candidates(problem_ids)

        # === bookkeeping ============================================= #
        n_probs  = len(problem_ids)
        n_rounds = max(len(cands) for cands in all_cands)
        syn_first: List[int | None] = [None] * n_probs
        sem_first: List[int | None] = [None] * n_probs

        # === candidate‑level evaluation + artefact saving ============== #
        for p_idx, (pid, cand_list) in enumerate(zip(problem_ids, all_cands)):
            for idx, cand in enumerate(cand_list):
                syn_ok, sem_ok, _, _ = self._check_candidate(pid, idx, cand, out_dir)

                if syn_ok and syn_first[p_idx] is None:
                    syn_first[p_idx] = idx
                if sem_ok and sem_first[p_idx] is None:
                    sem_first[p_idx] = idx

                if syn_first[p_idx] is not None and sem_first[p_idx] is not None:
                    break  # early exit for this problem

        # === round‑level accuracies =================================== #
        syn_acc_r: List[float] = []
        sem_acc_r: List[float] = []
        for r in range(n_rounds):
            syn_acc_r.append(sum(1 for s in syn_first if s is not None and s <= r) / n_probs)
            sem_acc_r.append(sum(1 for s in sem_first if s is not None and s <= r) / n_probs)

        # === save CSV ================================================= #
        pd.DataFrame(
            {
                "round": list(range(n_rounds)),
                "syntactic_accuracy": syn_acc_r,
                "semantic_accuracy": sem_acc_r,
            }
        ).to_csv(os.path.join(out_dir, "round_metrics.csv"), index=False)

        # === return final (latest‑round) numbers ====================== #
        return syn_acc_r[-1], sem_acc_r[-1]
