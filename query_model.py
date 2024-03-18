import csv
import torch
import string
import argparse
import warnings
import pandas as pd
from time import time
from tqdm import tqdm
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer, \
        AutoModelForSeq2SeqLM, pipeline, set_seed
from sentence_transformers import SentenceTransformer, util

from prompt_structures import ALL_PROMPTS

def get_args():
    """ Get arguments from command line. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='google/t5-xl-lm-adapt', 
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
    parser.add_argument('--match_type', type=str, default='exact', choices=['exact', 'fuzzy'],
                        help='Type of matching to use for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help='Verbosity level')
    args = parser.parse_args()

    return args


def prompt_from_3tuple(target, source, target_word):
    return f"If {target} is like {source}, then {target_word} is like"


def prompt_from_prompt_structure(ps, target, source, target_word):
    return ps.replace('_t_', target).replace('_s_', source).replace('_tw_', target_word)


def baseline(target_word, source_word, source):
    return [target_word, source_word, source]


def pad_outputs(beam_outputs, num_beams=10):
    """ Pad beam outputs to num_beams. """

    if len(beam_outputs) < num_beams:
        beam_outputs += [''] * (num_beams - len(beam_outputs))
    return beam_outputs


def remove_punctuation(text):
    """ Remove punctuation from text. """
    return text.translate(str.maketrans('', '', string.punctuation))


class Evaluator:
    def __init__(self, model_name, num_beams=10, sent_trans_model='all-MiniLM-L6-v2', 
                 match_type='exact', match_threshold=0.8):
        self.model_name = model_name
        self.num_beams = num_beams
        self.sent_trans_model = sent_trans_model
        self.match_threshold = match_threshold
        self.match_type = match_type

        if self.match_type == 'fuzzy':
            self.match_model = SentenceTransformer(sent_trans_model)
        else:
            self.match_model = None

        self.match_fn = self._fuzzy_match if self.match_type == 'fuzzy' else self._exact_match

    def _exact_match(self, output, targets):
        return any([x in output for x in targets])
    
    def _fuzzy_match(self, output, targets):
        out_tokens = output.lower().split(' ')
        sims = util.cos_sim(self.match_model.encode(out_tokens), self.match_model.encode(targets))
        return sims.max() > self.match_threshold
    
    def _mean_reciprocal_rank(self, rank_list):
        """ Compute mean reciprocal rank of the rank list. """
        return sum([1 / x for x in rank_list]) / len(rank_list)

    def get_rank(self, beam_outputs, targets):
        """ Get rank of the target word(s) in the beam outputs. """

        for i, beam_output in enumerate(beam_outputs):
            if self.match_fn(beam_output, targets):
                return i + 1
        
        return self.num_beams
    
    def get_acc(self, beam_outputs, targets):
        """ Return 1 if any target word are in the beam outputs. """

        for i, beam_output in enumerate(beam_outputs):
            if self.match_fn(beam_output, targets):
                return 1
        
        return 0

    def calc_metrics(self, outputs, targets):
        """ Calculate metrics. """

        rank_list, acc_list = [], []
        for beam_outputs, target_alts in tqdm(zip(outputs, targets), total=len(outputs)):
            rank_list.append(self.get_rank(beam_outputs, target_alts))
            acc_list.append(self.get_acc(beam_outputs, target_alts))

        mrr = self._mean_reciprocal_rank(rank_list)
        acc = sum(acc_list) / len(acc_list)
        
        return {'mrr': round(mrr*100, 3), 'acc': round(acc*100, 3)}
    

def load_model_and_tokenizer(model_name, device):
    """ Load model and tokenizer. """

    if model_name == 'baseline':
        model_dict = {'model': baseline}
    elif 't0' in model_name.lower() or 't5' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        model_dict = {'model': model, 'tokenizer': tokenizer}
    elif model_name == 'gpt2':
        device = -1 if device == 'cpu' else 0
        generator = pipeline('text-generation', model=model_name, device=int(device))
        model_dict = {'generator': generator}
    elif 'dialogpt' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model_dict = {'model': model, 'tokenizer': tokenizer}
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model_dict = {'model': model, 'tokenizer': tokenizer}

    return model_dict

def generate_beam_outputs(args, prompt, target, target_word, source, model_dict):
    if args.model == 'baseline':
        beam_outputs = model_dict['model'](target, target_word, source)
    elif args.model.lower() == 'gpt2':
        beam_outputs = model_dict['generator'](prompt, 
                                               max_length=40, 
                                               num_return_sequences=args.num_beams,
                                               pad_token_id=50256)
        beam_outputs = [x['generated_text'].lower()[len(prompt):].replace('"', '').strip() 
                        for x in beam_outputs]
    else:
        # Encode prompt
        if 'dialogpt' in args.model.lower() or 'gpt-neo' in args.model.lower():
            input_ids = model_dict['tokenizer'].encode(
                    model_dict['tokenizer'].eos_token + prompt, return_tensors='pt'
                ).to(args.device)
        else:
            input_ids = model_dict['tokenizer'].encode(
                    prompt + model_dict['tokenizer'].eos_token, return_tensors='pt'
                ).to(args.device)

        # Generate beam outputs
        generated_ids = model_dict['model'].generate(input_ids, 
                                                     max_length=40, 
                                                     num_beams=args.num_beams, 
                                                     num_return_sequences=args.num_beams,
                                                     pad_token_id=model_dict['tokenizer'].eos_token_id,
                                                     early_stopping=True).detach().cpu()

        # Decode beam outputs
        beam_outputs = [model_dict['tokenizer'].decode(x, skip_special_tokens=True).lower().replace('"', '') 
                        for x in generated_ids]
        
    return beam_outputs


def save_outputs(args, prompts, outputs, targets):
    """ Save outputs to file. """

    with open("data/scan_" + remove_punctuation(args.model) + "_outputs.txt", "w") as f:
        for i in tqdm(range(len(prompts))):
            f.write(prompts[i] + "\n" + str(outputs[i]) + "\n" + str(targets[i]) + "\n\n")
            

def main(args):
    """ Main function. """
    set_seed(args.seed)

    print(f"Device: {args.device}, Cuda is available: {torch.cuda.is_available()}.")

    t0 = time()

    # Load model and tokenizer
    model_dict = load_model_and_tokenizer(args.model, args.device)

    if args.verbose > 0:
        print(f"Loaded model {args.model} in {round(time() - t0, 3)} seconds to device '{args.device}'.\n")
        
    n_prompts = len(ALL_PROMPTS)
    all_metrics = []

    for i, prompt_structure in enumerate(ALL_PROMPTS):
        t1 = time()
        if args.verbose > 0:
            print(f"Prompt [{i+1} of {n_prompts}]: {prompt_from_prompt_structure(prompt_structure, 'target', 'source', 'target_word')}")

        prompts, outputs, targets = [], [], []

        # Read data
        with open('data/scan.csv', newline='') as f:
            reader = csv.reader(f, delimiter=',', quotechar='/')
            for i, row in enumerate(tqdm(reader)):
                # Skip header
                if i == 0: continue

                target, source, target_word, source_word = row[:4]
                prompt = prompt_from_prompt_structure(prompt_structure, target, source, target_word)

                # Process alternatives
                alternatives = row[4:-1]
                alternatives = [x.strip().replace('"', '') for x in alternatives if x.strip() != '']

                # Generate beam outputs
                beam_outputs = generate_beam_outputs(args, prompt, target, target_word, source, model_dict)

                if args.verbose > 1:
                    print(f"Prompt: {prompt}")
                    print(f"Alternatives: {[source_word] + alternatives}")
                    print(f"Beam outputs: {beam_outputs}\n")

                # Store outputs and targets
                prompts.append(prompt)
                outputs.append(pad_outputs(beam_outputs, num_beams=args.num_beams))
                targets.append([source_word] + alternatives)

        save_outputs(args, prompts, outputs, targets)

        evaluator = Evaluator(args.model, num_beams=args.num_beams, match_type=args.match_type)
        metrics = evaluator.calc_metrics(outputs, targets)
        all_metrics.append(
            [prompt_from_prompt_structure(prompt_structure, 'target', 'source', 'target_word')] \
            + list(metrics.values())
        )

        if args.verbose > 0:
            print("Metrics: {}".format(args.model, metrics))
            print(f"Finished in {round(time() - t1, 3)} seconds.\n")

    df_metrics = pd.DataFrame(all_metrics, columns=['prompt', 'mrr', 'acc'])

    if args.verbose > 0:
        print(f"Evaluated model {args.model}. Finished in {round(time() - t0, 3)} seconds.")
        print(tabulate(df_metrics, headers = 'keys', tablefmt = 'psql'))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = get_args()
    main(args)