"""
BestOfAllInference – pass@N + summary→PDDL + repair (solver+VAL)
saves per-round syntactic / semantic in metrics_best_of_all.csv
"""
from __future__ import annotations
from typing import List, Dict
from vllm import SamplingParams
import csv, os

from .summary_pddl_inference import SummaryPDDLInference

class BestOfAllInference(SummaryPDDLInference):
    def __init__(self,*a,n:int=8,max_rounds:int=4,**kw):
        super().__init__(*a,**kw); self.n=n; self.r=max_rounds
    # --------------------------------------------- #
    def batch_generate_candidates(self,pids:List[str])->List[List[Dict]]:
        # summary prompt batching from parent
        cands0=[r[0] for r in super().batch_generate_candidates(pids)]
        # repeat each prompt for pass@N
        prom=[self.get_prompt(pid) for pid in pids]
        sp=SamplingParams(temperature=self.temperature,top_p=0.95,max_tokens=20_000,n=self.n)
        outs=self.llm.generate(prom,sp)
        attempts=[[self._resp2dict(o.text) for o in item.outputs] for item in outs]
        per_round=[]
        for rnd in range(self.r+1):
            syn=sem=0
            for pid,alist in zip(pids,attempts):
                ok_syn=ok_sem=False
                for cand in alist:
                    s_ok,se_ok,_,_=self._check_candidate(pid,0,cand,"/tmp")
                    ok_syn|=s_ok; ok_sem|=se_ok
                    if ok_sem: break
                syn+=ok_syn; sem+=ok_sem
            per_round.append({"round":rnd,"syntactic":syn/len(pids),"semantic":sem/len(pids)})
            if rnd==self.r: break
            # build feedback prompts
            rev_prompts=[]; idx_map=[]
            for i,(pid,alist) in enumerate(zip(pids,attempts)):
                for cand in alist:
                    s_ok,se_ok,sol_err,val_err=self._check_candidate(pid,0,cand,"/tmp")
                    if se_ok: break
                    fb=sol_err if not s_ok else val_err
                    rev_prompts.append(self.get_prompt(pid)+"\n\n"+cand["raw"]+
                                       "\n\nFeedback:\n"+fb+"\n<think>")
                    idx_map.append(i); break
            if not rev_prompts: break
            new_out=self.llm.generate(rev_prompts,self.sampler)
            for o,i in zip(new_out,idx_map):
                attempts[i].append(self._resp2dict(o.outputs[0].text))
        # save per-round csv
        out_dir=os.path.join("output_tmp","metrics_bo"); os.makedirs(out_dir,exist_ok=True)
        with open(os.path.join(out_dir,"metrics_best_of_all.csv"),"w",newline="") as f:
            csv.DictWriter(f,fieldnames=per_round[0].keys()).writerows(per_round)
        return attempts
    def get_prompt(self,pid): return super().get_prompt(pid)
