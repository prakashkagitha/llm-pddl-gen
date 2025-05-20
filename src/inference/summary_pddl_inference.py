import os, re
from typing import List, Dict
from .base_inference import BaseInference

class SummaryPDDLInference(BaseInference):
    def _prefix(self):
        with open("prompts/only_pddl_instruction.txt") as f: return f.read().rstrip()+"\n\n"
    def _descs(self,pid):
        base=os.path.join("data",f"textual_{self.domain}",self.data_type)
        with open(os.path.join(base,f"{pid}_domain.txt")) as f: d=f.read()
        with open(os.path.join(base,f"{pid}_problem.txt")) as f: p=f.read()
        return d,p
    def batch_generate_candidates(self,pids:List[str])->List[List[Dict]]:
        sum_prompts=[]
        doms,probs=[],[]
        for pid in pids:
            d,p=self._descs(pid);doms.append(d);probs.append(p)
            sum_prompts.append(self._prefix()
            + "Domain description:\n" + d + "\n\nProblem description:\n" + p
            + "Write natural language PDDL summary needed to generate the domain and problem files in minimal PDDL."
            "Given the natural‑language descriptions below, produce a bullet‑point\n"
            "summary that lists *all* of the following:\n"
            "  • object types and concrete objects\n"
            "  • every predicate with its argument types\n"
            "  • every action/operator (name, parameters, pre‑conditions, effects)\n"
            "  • the initial state facts for the specific problem\n"
            "  • the goal condition\n"
            "\n"
            """
            <pddl_summary>
            - **Object Types and Concrete Objects:**
            - Types: ...
            - Objects: ...

            - **Predicates:**
            - ...

            - **Actions:**
            - ...
            - **Initial State Facts:**
            - ...

            - **Goal Condition:**
            - ...
            </pddl_summary>
            """
            + "\n\nTASK: Write the summary inside <pddl_summary> … </pddl_summary>. Do not\n"
            "output PDDL yet. Start your reasoning with <think>.\n")
        sums=[self._unwrap(o.outputs[0].text,"pddl_summary")
              for o in self.llm.generate(sum_prompts,self.sampler)]
        pddl_prompts=[self._prefix()
                      +"You are an expert PDDL engineer.\n"
            "Use ONLY the information provided below: a structured summary, the\n"
            "original domain description, and the original problem description.\n"
            "Ensure consistency between domain and problem.\n"
            "\n"
            "\n"
            "### Original domain description\n"
            f"{d}\n\n"
            "### Original problem description\n"
            f"{p}\n\n"
            "### Structured summary\n"
            f"{s}\n\n"
            "Write PDDL without any syntax errors that solves for the above "
            "domain and problem description with the help of summary"
            "Wrap *only* the domain file in <domain_file> … </domain_file> and\n"
            "the problem file in <problem_file> … </problem_file>.  Do not add\n"
            "any text inside those tags.\n<think>"
                      for s,d,p in zip(sums,doms,probs)]
        outs=self.llm.generate(pddl_prompts,self.sampler)
        return [[self._resp2dict(o.outputs[0].text,summary=s)]
                for o,s in zip(outs,sums)]
    def get_prompt(self,pid): raise NotImplementedError
