from .pddl_knowledge_inference import PDDLKnowledgeInference

class PassNInference(PDDLKnowledgeInference):
    def __init__(self, *a, n: int = 8, **kw): super().__init__(*a, **kw); self.n = n
    def batch_generate_candidates(self, pids):
        return self.get_multiple_batch_responses(pids, self.n)
