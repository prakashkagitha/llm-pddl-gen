import os
from .base_inference import BaseInference

class PDDLKnowledgeInference(BaseInference):
    def _prefix(self):
        with open("prompts/pddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    def _descs(self, pid):
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f: dom = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f: prob = f.read()
        return dom, prob

    def get_prompt(self, pid):
        dom, prob = self._descs(pid)
        return (
            self._prefix()
            + "Domain description:\n" + dom + "\n\nProblem description:\n" + prob
            + "Write the domain and problem files in minimal PDDL."
            + "\n\nWrap PDDL domain file inside <domain_file>…</domain_file> and PDDL problem file inside "
            "<problem_file>…</problem_file>.\n<think>"
        )