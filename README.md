# Unifying Inference-Time Planning Language Generation

Paper: [https://arxiv.org/abs/2505.14763](https://arxiv.org/abs/2505.14763)

Prabhu P. Kagitha, Bo Sun, Ishan Desai, Andrew Zhu, Cassie Huang, Manling Li, Ziyang Li, Li Zhang

This repository is a minimal, complete snapshot of the code used in the paper. It pulls the core experiment runner, prompts, and inference/solver/utility/validation modules from the full `planning-llms` codebase while excluding caches and large experiment outputs.

## Repository layout
- `configs/`: experiment configuration YAMLs
- `data/`: textual datasets (e.g., `textual_<domain>`)
- `experiments/`: experiment entry points
- `prompts/`: prompt templates used by the pipelines
- `src/inference/`: pipeline implementations
- `src/solver/`: planning solver wrappers
- `src/utils/`: experiment utilities and helpers
- `src/validation/`: PDDL validation helpers

## Requirements
Python 3.10+ and the following packages/tools:
- `vllm`
- `PyYAML`
- `requests`
- `outlines` (only for constrained decoding experiments)
- `planutils` (only if using the offline `dual-bfws-ffparser` solver)
- The online solver path uses the planning.domains API and requires network access

## Running experiments from the paper

All experiments are configured through the YAML in `configs/`. After setting the configuration (including `tensor_parallel` for your GPU count), run:

python -m experiments.run_experiments --config configs/experiment_config.yaml

This repo intentionally omits generated outputs and caches; see `.gitignore` for ignored artifacts.

## Citation

```
@misc{kagitha2025unifyinginference,
      title={Unifying Inference-Time Planning Language Generation},
      author={Prabhu P. Kagitha and Bo Sun and Ishan Desai and Andrew Zhu and Cassie Huang and Manling Li and Ziyang Li and Li Zhang},
      year={2025},
      eprint={2505.14763},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14763}, 
}
```
