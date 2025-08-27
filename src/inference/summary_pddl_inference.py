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
            + "Write natural language reasoning needed to generate the domain and problem files in minimal PDDL."
            "Given the natural‑language descriptions below, generate a thorough\n"
            "reasoning that thinks about *all* of the following:\n"
            "  • object types and concrete objects\n"
            "  • every predicate with its argument types\n"
            "  • every action/operator (name, parameters, pre‑conditions, effects)\n"
            "  • the initial state facts for the specific problem\n"
            "  • the goal condition\n"
            "\n"
            """
            * * Reason about actions in the environment and their pre-conditions and effects.
            Think and reason about the environment like it you are an embodied agent in that environment.
            Think about all the minute details about the environment. Because the PDDL code 
            or python env code that you write should be comprehensive to create a virtual environment to a embodied agent.
            Check consistency between all the action semantics towards the domain and problem description.
            Self-critique your reasoning and ensure it is comprehensive.
            Self-critique about whether action semantics consider all the aspects of the environment.
            ** If needed, generate bespoke python wrapper code that could be translated to PDDL easily.
            ** If needed, generate python environment code that defines the environment, objects, 
            predicates, actions, initial state, and goal condition.
            Example output:
            <environment_reasoning>
            # thorough reasoning about different components of PDDL as mentioned above.
            ...
            # python environment code including action pre-conditions and post-conditions
            ...
            # Self-critique and revision of the above reasoning and python environnment
                whether it satisfies all the details from domain description and problem description
            ...
            # any other reasoning or cautions or example PDDL for this environment
            ...
            </environment_reasoning>
            """
            + "\n\nTASK: Write the reasoning summary that you reasoned through and reflected or self-critiqued about.\n"
            "including the python code that you may have written inside <environment_reasoning> … </environment_reasoning>."
            " We are going to use this reasoning to generate PDDL in the next step. So, be as thorough as possible."
            "Make <environment_reasoning> as vivid as possible so that this becomes a simulation for embodied agent. Do not\n"
            "output PDDL yet. Start your reasoning with <think>.\n")
        sums=[self._unwrap(o.outputs[0].text,"environment_reasoning")
              for o in self.llm.generate(sum_prompts,self.sampler)]
        pddl_prompts=[self._prefix()
                      +"You are an expert PDDL engineer.\n"
            "Use the information provided below: a structured summary, the\n"
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
