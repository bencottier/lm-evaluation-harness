import subprocess
import json
import os
from datetime import datetime

RESULTS_DIR = "output"


MODELS = [
    #"EleutherAI/gpt-j-6b",
    #"EleutherAI/gpt-neo-1.3B",
    # "EleutherAI/gpt-neo-125m",
    #"EleutherAI/gpt-neo-2.7B",
    #"EleutherAI/gpt-neox-20b",
    #"EleutherAI/pythia-1.4b",
    #"EleutherAI/pythia-12b",
    # "EleutherAI/pythia-160m",
    #"EleutherAI/pythia-1b",
    #"EleutherAI/pythia-2.8b",
    # "EleutherAI/pythia-410m",
    #"EleutherAI/pythia-6.9b",
    # "EleutherAI/pythia-70m",
    
    # could not get to work, could be a versioning issue
    #"THUDM/glm-10b,trust_remote_code=True",
    #"THUDM/glm-2b,trust_remote_code=True",
    
    #"bigscience/bloom-1b1",
    #"bigscience/bloom-1b7",
    #"bigscience/bloom-3b",
    # "bigscience/bloom-560m",
    #"bigscience/bloom-7b1",
    
    # https://github.com/huggingface/transformers/issues/22222#issuecomment-1525580655
    # "decapoda-research/llama-13b-hf",
    # "decapoda-research/llama-30b-hf",
    # "decapoda-research/llama-65b-hf",
    # "decapoda-research/llama-7b-hf",
    
    # more Llama, to double check
    #"elinas/llama-7b-hf-transformers-4.29",
    #"elinas/llama-13b-hf-transformers-4.29",
    #"elinas/llama-30b-hf-transformers-4.29",
    #"elinas/llama-65b-hf-transformers-4.29",
    
    #"facebook/opt-1.3b",
    # "facebook/opt-125m",
    #"facebook/opt-13b",
    #"facebook/opt-2.7b",
    #"facebook/opt-30b",
    # "facebook/opt-350m",
    #"facebook/opt-6.7b",
    #"facebook/opt-66b",
    #"huggyllama/llama-13b",
    #"huggyllama/llama-30b",
    #"huggyllama/llama-7b",
    #"mosaicml/mpt-7b,trust_remote_code=True",

    "cerebras/Cerebras-GPT-111M",
    
    # "distilgpt2",
]

TASKS = ",".join(['wikitext', 'penn_treebank', 'lambada_standard', 'mnli'])


if __name__ == '__main__':
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    for model in MODELS:
        print(f"{datetime.now().isoformat()}: running {model} on {TASKS}")
        model_fname = model.replace('/', '_')
        try:
            subprocess.run(['python3', 'main.py', '--model', 'hf-causal-experimental',
                            '--model_args', f'pretrained={model},use_accelerate=True', '--tasks',
                            TASKS, '--batch_size', 'auto', '--device', 'cuda:0', '--write_out', '--output_path',
                            f'{RESULTS_DIR}/{model_fname}.json'], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            t_error = datetime.now().isoformat()
            print(f"{t_error}: ERROR {model} on {TASKS}")
            with open(f'{RESULTS_DIR}/{model_fname}_errors.log', 'w') as f:
                f.write(f'Time: {t_error}\n')
                f.write(f'Model: {model}\n')
                f.write('\n==== STDOUT ====\n\n')
                f.write(e.stdout.decode('utf-8'))
                f.write('\n==== STDERR ====\n\n')
                f.write(e.stderr.decode('utf-8'))
    # combine results into single file
    results = []
    for output_fname in os.listdir(RESULTS_DIR):
        if "errors.log" not in output_fname and not os.path.isdir(output_fname):
            with open(f'{RESULTS_DIR}/{output_fname}') as f:
                results.append(json.load(f))
    with open(f'{RESULTS_DIR}/all_results.json', 'w') as f:
        json.dump(results, f, indent=4)