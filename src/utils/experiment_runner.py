"""
ExperimentRunner
================
Ultra-thin orchestration: it simply instantiates the requested pipeline
class, calls *evaluate()* to obtain (syntax_acc, sem_acc), writes a CSV
row, then frees GPU memory.
"""
import os
import pandas as pd
from importlib import import_module
from typing import Dict, List

from src.utils.file_manager import get_output_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------- #
class ExperimentRunner:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rows: List[Dict] = []

    # ------------------------------------------------------------------ #
    def _resolve_pipeline(self, name: str):
        mapping = {
            "baseline_pddl":        "baseline_pddl_inference.BaselinePDDLInference",
            "pddl_knowledge":       "pddl_knowledge_inference.PDDLKnowledgeInference",
            "separate_pddl":        "separate_pddl_inference.SeparatePDDLInference",
            "summary_pddl":         "summary_pddl_inference.SummaryPDDLInference",
            "pass_at_n":            "passn_inference.PassNInference",
            "always_revise":        "always_revise_inference.AlwaysReviseInference",
            "revision_solver":      "revision_solver_inference.RevisionSolverInference",
            "revision_solver_val":  "revision_solver_val_inference.RevisionSolverValInference",
            "pypddl":               "pypddl_inference.PyPDDLInference",
            "constrained_decoding": "constrained_decoding_inference.ConstrainedDecodingInference",
            "best_of_all":          "best_of_all_inference.BestOfAllInference",
        }

        modname, clsname = mapping[name].rsplit(".", 1)
        return getattr(import_module(f"src.inference.{modname}"), clsname)

    # ------------------------------------------------------------------ #
    def run(self):
        for model in self.cfg["llm_models"]:
            for domain, data in zip(self.cfg["domains"], self.cfg["data_types"]):
                for temp in self.cfg["temperatures"]:
                    for pipe in self.cfg["pipelines"]:
                        logger.info("▶ %s | %s | %s | %.2f", model, domain, pipe, temp)

                        out_dir = get_output_path(
                            model, self.cfg["prompt_versions"][0],
                            data, domain, temp, pipe
                        )
                        os.makedirs(out_dir, exist_ok=True)

                        PipelineCls = self._resolve_pipeline(pipe)
                        pipeline = PipelineCls(
                            model, temp, self.cfg["prompt_versions"][0],
                            domain, data,
                            tensor_parallel=self.cfg.get("tensor_parallel", 1),
                            n=self.cfg.get("pass_at_n", 8),                # only used by PassN
                            k=self.cfg.get("num_pass_attempts", 16),       # RevSolverVal
                            max_rounds=self.cfg.get("num_revision_rounds", 4),
                        )

                        problems = [f"p{p:02}" for p in self.cfg["problems"]]
                        syn_acc, sem_acc = pipeline.evaluate(problems, out_dir)
                        pipeline.close()

                        row = {
                            "model": model,
                            "domain": domain,
                            "data_type": data,
                            "temperature": temp,
                            "pipeline": pipe,
                            "syntactic_accuracy": syn_acc,
                            "semantic_accuracy": sem_acc,
                            "n_problems": len(problems),
                        }
                        self.rows.append(row)

                        # one CSV per combination
                        pd.DataFrame([row]).to_csv(
                            os.path.join(out_dir, "metrics.csv"), index=False
                        )

        # master CSV
        pd.DataFrame(self.rows).to_csv(
            f"{out_dir}/master_metrics.csv", index=False
        )
        logger.info(f"✅  finished – results in {out_dir}/master_metrics.csv")
