"""
AlwaysReviseInference – records per-round transition metrics.
"""
from vllm import SamplingParams
from .pddl_knowledge_inference import PDDLKnowledgeInference
import csv, os

class AlwaysReviseInference(PDDLKnowledgeInference):
    def __init__(self,*a,max_rounds:int=3,**kw):
        super().__init__(*a,**kw); self.r=max_rounds

    def batch_generate_candidates(self,pids):
        resps=self.get_batch_responses(pids)
        transitions=[]
        for rnd in range(self.r):
            syn0=[self._check_candidate(pid,0,r,"/tmp")[0] for pid,r in zip(pids,resps)]
            sem0=[self._check_candidate(pid,0,r,"/tmp")[1] for pid,r in zip(pids,resps)]
            rev_prompts=[self.get_prompt(pid)+"\n\n"+r["raw"]+"\n\nFix.\n<think>"
                         for pid,r in zip(pids,resps)]
            resps=[self._resp2dict(o.outputs[0].text)
                   for o in self.llm.generate(rev_prompts,self.sampler)]
            syn1=[self._check_candidate(pid,0,r,"/tmp")[0] for pid,r in zip(pids,resps)]
            sem1=[self._check_candidate(pid,0,r,"/tmp")[1] for pid,r in zip(pids,resps)]
            transitions.append({
                "round":rnd+1,
                "correct_to_incorrect":sum(1 for a,b in zip(sem0,sem1) if a and not b),
                "incorrect_to_correct":sum(1 for a,b in zip(sem0,sem1) if not a and b),
            })
        # save transition csv
        out_dir=os.path.join("output_tmp","metrics_ai"); os.makedirs(out_dir,exist_ok=True)
        if transitions:
            with open(os.path.join(out_dir,"metrics_always_revise.csv"),"w",newline="") as f:
                csv.DictWriter(f,fieldnames=transitions[0].keys()).writerows(transitions)
        return [[r] for r in resps]
