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
        # Add default seeds if not provided
        self.seeds = cfg.get("seeds", [42, 123, 456])  # Default 3 seeds
        self.system_prompt: str | None = cfg.get("system_prompt")

    # ------------------------------------------------------------------ #
    def _resolve_pipeline(self, name: str):
        mapping = {
            "baseline_pddl":        "baseline_pddl_inference.BaselinePDDLInference",
            "pddl_knowledge":       "pddl_knowledge_inference.PDDLKnowledgeInference",
            "pddl":                 "pddl_inference.PDDLInference",
            "separate_pddl":        "separate_pddl_inference.SeparatePDDLInference",
            "summary_pddl":         "summary_pddl_inference.SummaryPDDLInference",
            "nl_pddl":              "nl_pddl_inference.NLPDDLInference",
            "pddl_pddl":           "pddl_pddl_inference.PDDLPDDLInference",
            "pddl_pddl_nofb":     "pddl_pddl_nofb_inference.PDDLPDDLNoFBInference",
            "nl_nl_pddl":          "nl_nl_pddl_inference.NLNLPDDLInference",
            "nl_pypddl_pddl":      "nl_pypddl_pddl_inference.NLPyPDDLPDDLInference",
            "pddl_pypddl_pddl":    "pddl_pypddl_pddl_inference.PDDLPyPDDLPDDLInference",
            "pddl_pddl_pddl":     "pddl_pddl_pddl_inference.PDDLPDDLPDDLInference",
            "pddl_pddl_pddl_nofb": "pddl_pddl_pddl_nofb_inference.PDDLPDDLPDDLNoFBInference",
            "pypddl_pypddl_pddl":  "pypddl_pypddl_pddl_inference.PyPDDLPyPDDLPDDLInference",
            "pypddl_pddl":        "pypddl_pddl_inference.PyPDDLPDDLInference",
            "pypddl_pddl_pddl":    "pypddl_pddl_pddl_inference.PyPDDLPDDLPDDLInference",
            "pythonsim_pddl":     "pythonsim_pddl_inference.PythonSimPDDLInference",
            "json_pddl":          "json_pddl_inference.JSONPDDLInference",
            "pddl_plan":           "pddl_plan_inference.PDDLPlanInference",
            "pass_at_n":            "passn_inference.PassNInference",
            "always_revise":        "always_revise_inference.AlwaysReviseInference",
            "revision_solver":      "revision_solver_inference.RevisionSolverInference",
            "revision_solver_val":  "revision_solver_val_inference.RevisionSolverValInference",
            "pypddl":               "pypddl_inference.PyPDDLInference",
            "constrained_decoding": "constrained_decoding_inference.ConstrainedDecodingInference",
            "best_of_all":          "best_of_all_inference.BestOfAllInference",
            "pddlwithpy2pddl": "pddlwithpy2pddlinference.PDDLWithPy2PDDLInference",
            "pddlwithpythonenv": "pddlwithpythonenv_inference.PDDLWithPythonEnvInference",
            "plan_gen":             "plan_gen_inference.PlanGenInference",
        }

        modname, clsname = mapping[name].rsplit(".", 1)
        return getattr(import_module(f"src.inference.{modname}"), clsname)

    # ------------------------------------------------------------------ #
    def run(self):
        for model in self.cfg["llm_models"]:
            for domain, data in zip(self.cfg["domains"], self.cfg["data_types"]):
                for temp in self.cfg["temperatures"]:
                    for pipe in self.cfg["pipelines"]:
                        for run_idx, seed in enumerate(self.seeds):
                            logger.info("▶ %s | %s | %s | %.2f | Run %d (seed=%d)", 
                                       model, domain, pipe, temp, run_idx + 1, seed)

                            out_dir = get_output_path(
                                model, self.cfg["prompt_versions"][0],
                                data, domain, temp, pipe
                            )
                            # Create run-specific subdirectory
                            run_out_dir = os.path.join(out_dir, f"run_{run_idx + 1}_seed_{seed}")
                            os.makedirs(run_out_dir, exist_ok=True)

                            PipelineCls = self._resolve_pipeline(pipe)
                            pipeline = PipelineCls(
                                model, temp, self.cfg["prompt_versions"][0],
                                domain, data,
                                tensor_parallel=self.cfg.get("tensor_parallel", 1),
                                n=self.cfg.get("pass_at_n", 8),
                                k=self.cfg.get("num_pass_attempts", 16),
                                max_rounds=self.cfg.get("num_revision_rounds", 4),
                                seed=seed,  # Pass seed to pipeline
                                system_prompt=self.system_prompt,
                            )

                            problems = [f"p{p:02}" for p in self.cfg["problems"]]
                            metrics = pipeline.evaluate(problems, run_out_dir)
                            pipeline.close()

                            row = {
                                "model": model,
                                "domain": domain,
                                "data_type": data,
                                "temperature": temp,
                                "pipeline": pipe,
                                "run": run_idx + 1,
                                "seed": seed,
                            }
                            metrics = dict(metrics)
                            metrics.setdefault("n_problems", len(problems))
                            row.update(metrics)
                            self.rows.append(row)

                            # Save individual run metrics
                            pd.DataFrame([row]).to_csv(
                                os.path.join(run_out_dir, "metrics.csv"), index=False
                            )

        # Master CSV with all runs
        master_df = pd.DataFrame(self.rows)
        master_output_path = os.path.join(
            os.path.dirname(out_dir), "master_metrics_all_runs.csv"
        )
        master_df.to_csv(master_output_path, index=False)
        
        # Also create aggregated statistics
        metric_columns = [
            'syntactic_accuracy',
            'semantic_accuracy',
            'plan_not_found_count',
            'plan_not_valid_count',
            'tokens_per_valid_plan',
            'total_tokens'
        ]

        agg_spec = {metric: ['mean', 'std'] for metric in metric_columns if metric in master_df.columns}
        agg_spec['n_problems'] = 'first'

        agg_df = (
            master_df
            .groupby(['model', 'domain', 'data_type', 'temperature', 'pipeline'])
            .agg(agg_spec)
            .round(4)
        )

        flattened_cols = []
        for col in agg_df.columns:
            if isinstance(col, tuple):
                flattened_cols.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
            else:
                flattened_cols.append(col)
        agg_df.columns = flattened_cols
        agg_df.reset_index().to_csv(
            os.path.join(os.path.dirname(out_dir), "aggregated_metrics.csv"), 
            index=False
        )
        
        logger.info(f"✅  finished – detailed results in {master_output_path}")
        logger.info(f"✅  aggregated results in {os.path.join(os.path.dirname(out_dir), 'aggregated_metrics.csv')}")
