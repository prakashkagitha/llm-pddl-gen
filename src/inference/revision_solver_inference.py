# src/inference/revision_solver_inference.py
"""
RevisionSolverInference – v2
----------------------------
* k initial candidates per problem.
* Every candidate is run through the solver; its **own** error message
  becomes the feedback for that exact candidate.
* Each unsolved candidate is revised independently in every round,
  so you may end up with many more than *k* attempts per problem.
* All responses / plans are saved with monotonically-increasing
  cand_{idx:02} indices (idx == generation order).
* After the full loop we compute **per-round** syntactic & semantic
  accuracies and write them to  `<out_dir>/round_metrics.csv`.
"""
from __future__ import annotations

import os, itertools, tempfile
from typing import List, Dict, Tuple

import pandas as pd
from vllm import SamplingParams

from .pddl_knowledge_inference import PDDLKnowledgeInference


class RevisionSolverInference(PDDLKnowledgeInference):
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *a,
        k: int = 8,
        max_rounds: int = 3,
        **kw,
    ):
        super().__init__(*a, **kw)
        self.k = k
        self.r = max_rounds
        # temp dir only for feedback collection (final artefacts are saved
        # later, inside evaluate -> _check_candidate(out_dir, …))
        self._tmp = tempfile.mkdtemp(prefix="revsolver_")

    # ------------------------------------------------------------------ #
    #  Generation-and-revision loop                                      #
    # ------------------------------------------------------------------ #
     # ------------------------------------------------------------------ #
    #  Generation-and-revision loop – *new*: only last-round candidates  #
    # ------------------------------------------------------------------ #
    def batch_generate_candidates(
        self,
        pids: List[str],
    ) -> List[List[Dict]]:
        """
        k initial candidates → round = 0  
        At round r ≥ 1 we **only** revise candidates whose ``_round == r-1``.
        Thus every candidate forms a *chain* of sequential revisions,
        and the prompt count can only stay flat or shrink.
        """
        # ---- round-0 --------------------------------------------------
        attempts: List[List[Dict]] = self.get_multiple_batch_responses(pids, self.k)
        for cand_list in attempts:                 # tag origin round
            for cand in cand_list:
                cand["_round"] = 0

        # ---- rounds 1 … R --------------------------------------------
        for cur_r in range(1, self.r + 1):
            rev_prompts: List[str] = []
            tgt_mapping: List[int] = []            # index into *attempts*

            # pick *only* candidates generated in the previous round
            for p_idx, (pid, cand_list) in enumerate(zip(pids, attempts)):
                for c_idx, cand in enumerate(cand_list):
                    if cand["_round"] != cur_r - 1:
                        continue                   # not a last-round candidate

                    syn_ok, sem_ok, err, _ = self._check_candidate(
                        pid, c_idx, cand, self._tmp
                    )
                    if sem_ok:
                        continue                   # solved → no further revision

                    rev_prompts.append(
                        self._build_revision_prompt(pid, cand, err)
                    )
                    tgt_mapping.append(p_idx)      # append to this problem

            # nothing left to revise → stop early
            if not rev_prompts:
                break

            # batch-generate all revisions for this round
            new_outs = self.llm.generate(rev_prompts, self.sampler)
            for p_idx, out in zip(tgt_mapping, new_outs):
                new_cand = self._resp2dict(out.outputs[0].text)
                new_cand["_round"] = cur_r        # tag creation round
                attempts[p_idx].append(new_cand)

        return attempts


    # -------------------- helpers ------------------------------------- #
    def _build_revision_prompt(self, pid: str, cand: Dict[str, str], feedback: str) -> str:
        """
        Creates an instruction that:
          1. shows the **exact** domain & problem PDDL produced earlier
          2. attaches the solver’s error message
          3. asks for a revised answer (with proper XML wrappers)
        """
        return (
            self.get_prompt(pid) +
            "------ Previous attempt start ------\n"
            f"[DOMAIN FILE]\n{cand['df']}\n\n"
            f"[PROBLEM FILE]\n{cand['pf']}\n"
            "------ Previous attempt end --------\n\n"
            f"Feedback from the planning solver:\n{feedback.strip()}\n\n"
            "Think step-by-step about what is wrong.  Produce a **corrected** "
            "answer.  Wrap the new domain file in <domain_file>…</domain_file> "
            "and the new problem file in <problem_file>…</problem_file>.\n"
            "<think>"
        )

    # ------------------------------------------------------------------ #
    #  Custom evaluate(): collects per-round metrics                     #
    # ------------------------------------------------------------------ #
    def evaluate(  # type: ignore[override]
        self,
        problem_ids: List[str],
        out_dir: str,
    ):
        """
        Returns final (syntactic_accuracy, semantic_accuracy) *and* writes
        per-round progress to  round_metrics.csv  inside *out_dir*.
        """
        # === generate / revise ========================================
        all_cands = self.batch_generate_candidates(problem_ids)

        # === metric bookkeeping =======================================
        n_probs     = len(problem_ids)
        n_rounds    = max(len(cand_list) for cand_list in all_cands)
        syn_first   = [None] * n_probs            # earliest round with syn-OK
        sem_first   = [None] * n_probs            # earliest round with sem-OK
        syn_acc_r   = [0] * n_rounds
        sem_acc_r   = [0] * n_rounds

        # === candidate-level evaluation + artefact saving =============
        for p_idx, (pid, cand_list) in enumerate(zip(problem_ids, all_cands)):
            for idx, cand in enumerate(cand_list):
                syn_ok, sem_ok, _, _ = self._check_candidate(pid, idx, cand, out_dir)

                if syn_ok and syn_first[p_idx] is None:
                    syn_first[p_idx] = idx
                if sem_ok and sem_first[p_idx] is None:
                    sem_first[p_idx] = idx

                # micro-optimisation: if both already solved → stop early
                if (
                    syn_first[p_idx] is not None
                    and sem_first[p_idx] is not None
                ):
                    # no need to check later candidates for this problem
                    break

        # === aggregate accuracies per round ===========================
        for r in range(n_rounds):
            for p in range(n_probs):
                if syn_first[p] is not None and syn_first[p] <= r:
                    syn_acc_r[r] += 1
                if sem_first[p] is not None and sem_first[p] <= r:
                    sem_acc_r[r] += 1

        syn_acc_r = [v / n_probs for v in syn_acc_r]
        sem_acc_r = [v / n_probs for v in sem_acc_r]

        # === save round-level CSV =====================================
        pd.DataFrame(
            {
                "round": list(range(n_rounds)),
                "syntactic_accuracy": syn_acc_r,
                "semantic_accuracy": sem_acc_r,
            }
        ).to_csv(os.path.join(out_dir, "round_metrics.csv"), index=False)

        # === return final numbers (latest round) ======================
        return syn_acc_r[-1], sem_acc_r[-1]
