# Addressing the Challenges of Planning Language Generation

Paper: [https://arxiv.org/abs/2505.14763](https://arxiv.org/abs/2505.14763)

Prabhu Prakash Kagitha (Drexel University), Andrew Zhu (University of Pennsylvania), Harry "Li" Zhang (Drexel Univeristy)

## Requirements
vLLM
outlines # only for running constrained decoding experiments

## Running experiments from the paper

All the experiments could be run based on the config YAML file  in configs. After a specific configuration is set, use the following command to run the experiments. (Adjust tensor_parallel based on the number of GPUs to run on)

python -m experiments.run_experiments --config configs/experiment_config.yaml

## Citation

```
@misc{kagitha2025addressingchallengesplanninglanguage,
      title={Addressing the Challenges of Planning Language Generation}, 
      author={Prabhu Prakash Kagitha and Andrew Zhu and Li Zhang},
      year={2025},
      eprint={2505.14763},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.14763}, 
}
```