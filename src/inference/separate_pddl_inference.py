import os, re
from functools import cached_property
from typing import List, Dict
from .base_inference import BaseInference

class SeparatePDDLInference(BaseInference):
    @cached_property
    def _prefix(self):
        with open("prompts/only_pddl_instruction.txt") as f: return f.read().rstrip()+"\n\n"

    def _descs(self, pid):
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base,f"{pid}_domain.txt")) as f: dom=f.read()
        with open(os.path.join(base,f"{pid}_problem.txt")) as f: prob=f.read()
        return dom, prob

    # ---------------- batch implementation ------------------- #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        dom_prompts = [self._prefix + d +
                       """
                       Write the domain file in minimal PDDL.
                       Wrap PDDL domain file inside <domain_file>…</domain_file>."""
                       + "\n<think>"
                       for d, _ in map(self._descs, pids)]
        dom_out = self.llm.generate(dom_prompts, self.sampler)
        dom_files = [self._unwrap(o.outputs[0].text, "domain_file") for o in dom_out]

        prob_prompts = [
            self._prefix + "\nDOMAIN DESC:\n" + self._descs(pid)[0] 
            + "\nPROBLEM DESC:\n" + self._descs(pid)[1]
            + "DOMAIN FILE already generated:\n" + df
            + """\n Write the problem file for the given problem description and domain file in minimal PDDL.
            Wrap PDDL problem file inside <problem_file>…</problem_file>."""
            + "\n<think>"
            for pid, df in zip(pids, dom_files)
        ]
        prob_out = self.llm.generate(prob_prompts, self.sampler)
        problem_files = [
            self._unwrap(o.outputs[0].text, "problem_file") for o in prob_out
        ]

        return [[{"df": d, "pf": p, "raw": ""}] for d, p in zip(dom_files, problem_files)]

    # dummy
    def get_prompt(self, pid): raise NotImplementedError
