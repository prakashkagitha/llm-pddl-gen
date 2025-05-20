import argparse
import yaml
import os
from src.utils.experiment_runner import ExperimentRunner
from src.utils.logger import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM planning experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML file")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    runner = ExperimentRunner(config)
    runner.run()
    
if __name__=="__main__":
    main()
