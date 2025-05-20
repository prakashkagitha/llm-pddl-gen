All the experiments could be run based on the config YAML file  in configs. After a specific configuration is set, use the following command to run the experiments. (Adjust tensor_parallel based on the number of GPUs to run on)

python -m experiments.run_experiments --config configs/experiment_config.yaml