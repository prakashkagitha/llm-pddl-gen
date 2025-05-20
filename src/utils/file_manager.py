import os

def get_input_path(data_type, problem, file_type):
    # file_type: "domain" or "problem"
    return os.path.join("data", data_type, f"{problem}_{file_type}.txt")

def get_output_path(model, prompt_version, data_type, domain, temperature, pipeline_option):
    # Create output folder structure based on all parameters.
    return os.path.join("output", "llm-as-formalizer", domain, data_type, model, pipeline_option, f"{prompt_version}_temp{temperature}")
