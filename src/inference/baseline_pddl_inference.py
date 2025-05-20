from .pddl_knowledge_inference import PDDLKnowledgeInference

class BaselinePDDLInference(PDDLKnowledgeInference):
    def _prefix(self): return ""  # removes knowledge prompt