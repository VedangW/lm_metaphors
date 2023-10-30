import csv
import torch
import argparse
import warnings
from time import time
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, \
        AutoModelForSeq2SeqLM, pipeline, set_seed
from utils import pad_outputs, calc_metrics


def get_args():
    """ Get arguments from command line. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-medium', 
                        choices=['microsoft/DialoGPT-medium', 
                                 'microsoft/DialoGPT-large', 
                                 'bigscience/bloom-7b1',
                                 'bigscience/T0_3B',
                                 'bigscience/T0',
                                 'bigscience/T0pp',
                                 'google/t5-xl-lm-adapt',
                                 'EleutherAI/gpt-neo-2.7B',
                                 'gpt2',
                                 'baseline'], 
                        help='Model to use for inference')
    parser.add_argument('--num_beams', type=int, default=10, 
                        help='Number of beams to use for beam search')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run the model on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help='Verbosity level')
    args = parser.parse_args()

    return args


def generate_in_context_prompt(target, data):
    for i in range(len(data)):
        prompt = ""
        for j in range(len(data)):
            if i != j:
                source, target_word, source_word, _ = data[j]
                prompt += f"If '{target}' is like '{source}', then '{target_word}' is like '{source_word}'. "

        prompt += f"If '{target}' is like '{data[i][0]}', then '{data[i][1]}' is like what? Answer in one word."

        yield prompt, data[i][2], data[i][3]


def main(args):
    set_seed(args.seed)

    t0 = time()

    # Load model and tokenizer
    if 't0' in args.model.lower() or 't5' in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
    elif 'gpt' in args.model:
        generator = pipeline('text-generation', model=args.model, device=int(args.device))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    if args.verbose > 0:
        print("Loaded model in {} seconds".format(round(time() - t0, 3)))

    domains = defaultdict(list)

    with open('data/scan.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='/')
        for i, row in enumerate(tqdm(reader)):
            if i == 0:
                continue
            
            target, source, target_word, source_word = row[:4]
            alternatives = row[4:-1]
            alternatives = [x.strip().replace('"', '') for x in alternatives if x.strip() != '']
            
            domains[target].append((source, target_word, source_word, alternatives))

    outputs, targets = [], []

    for target, data in domains.items():
        print(f"Target: {target}.")
        for prompt, source_word, alternatives in tqdm(generate_in_context_prompt(target, data)):
            if args.model == 'baseline':
                beam_outputs = model(target, target_word, source)
            elif 'gpt' in args.model.lower():
                beam_outputs = generator(prompt, 
                                            max_length=500, 
                                            num_return_sequences=args.num_beams,
                                            pad_token_id=50256)
                beam_outputs = [x['generated_text'].lower()[len(prompt):].replace('"', '').strip() 
                                for x in beam_outputs]
            else:
                # Encode prompt
                input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt').to(args.device)

                # Generate beam outputs
                generated_ids = model.generate(input_ids, 
                                                max_length=40, 
                                                num_beams=args.num_beams, 
                                                num_return_sequences=args.num_beams,
                                                pad_token_id=tokenizer.eos_token_id,
                                                early_stopping=True).detach().cpu()

                # Decode beam outputs
                beam_outputs = [tokenizer.decode(x, skip_special_tokens=True).lower().replace('"', '') 
                                for x in generated_ids]

            if args.verbose > 1:
                print(f"Prompt: {prompt}")
                print(f"Alternatives: {[source_word] + alternatives}")
                print(f"Beam outputs: {beam_outputs}\n")

            # Store outputs and targets
            outputs.append(pad_outputs(beam_outputs, num_beams=args.num_beams))
            targets.append([source_word] + alternatives)

    if args.verbose > 0:
        print("Model: {}, Metrics: {}".format(args.model, calc_metrics(outputs, targets, num_beams=args.num_beams)))
        print(f"Finished in {round(time() - t0, 3)} seconds.")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = get_args()
    main(args)
