import os
from typing import List, Dict
from .base_inference import BaseInference


class PDDLWithPy2PDDLInference(BaseInference):
    # ------------------------------------------------------------------ #
    # prompt prefixes
    def _py_prefix(self) -> str:
        with open("prompts/pypddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    def _pddl_prefix(self) -> str:
        with open("prompts/only_pddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    # ------------------------------------------------------------------ #
    def _descs(self, pid: str):
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f:
            d = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f:
            p = f.read()
        return d, p

    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        # ---------- STEP 1  NL → py2pddl Python ----------
        py_prompts, doms, probs = [], [], []
        for pid in pids:
            d_desc, p_desc = self._descs(pid)
            doms.append(d_desc), probs.append(p_desc)

            py_prompts.append(
                self._py_prefix()
                + "### Domain description\n" + d_desc
                + "\n\n### Problem description\n" + p_desc
                + "\n\n"
                  "TASK: Write executable Python 3 using **py2pddl** that models the "
                  "domain and problem above.  Wrap ONLY the domain class in "
                  "<domain_file>…</domain_file> and the problem class in "
                  "<problem_file>…</problem_file>.  No extra text.\n<think>"
            )

        py_outs = self.llm.generate(py_prompts, self.sampler)
        py_domains, py_problems = [], []
        for o in py_outs:
            txt = o.outputs[0].text.split("</think>", 1)[-1]
            py_domains.append(self._unwrap(txt, "domain_file"))
            py_problems.append(self._unwrap(txt, "problem_file"))

        # ---------- STEP 2  (py2pddl + NL) → final PDDL ----------
        pddl_prompts = []
        for d_desc, p_desc, py_dom, py_prob in zip(doms, probs, py_domains, py_problems):
            pddl_prompts.append(
                self._pddl_prefix()
                + "You are an expert PDDL engineer.\n"
                  "Using ONLY the information below, produce syntactically correct and "
                  "mutually consistent PDDL.\n\n"
                + "### Original domain description\n" + d_desc
                + "\n\n### Original problem description\n" + p_desc
                + "\n\n### py2pddl domain class\n```python\n" + py_dom.strip() + "\n```\n"
                + "\n### py2pddl problem class\n```python\n" + py_prob.strip() + "\n```\n\n"
                  "Wrap *only* the final domain PDDL in <domain_file>…</domain_file> and "
                  "the final problem PDDL in <problem_file>…</problem_file>. "
                  "No additional text.\n<think>"
            )

        pddl_outs = self.llm.generate(pddl_prompts, self.sampler)

        # ---------- package results ----------
        candidates = []
        for out, py_dom, py_prob in zip(pddl_outs, py_domains, py_problems):
            cand = self._resp2dict(out.outputs[0].text)      # gives df, pf, raw, summary
            cand["python_domain_file"] = py_dom              # keep intermediate artefacts
            cand["python_problem_file"] = py_prob
            candidates.append([cand])

        return candidates

    def get_prompt(self, pid: str):
        raise NotImplementedError