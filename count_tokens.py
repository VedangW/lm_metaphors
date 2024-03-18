import argparse
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='data/scan.csv', help='file to count tokens in')
parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='model to use')
parser.add_argument('--out_tokens', type=int, default=20, help='number of tokens to output')
args = parser.parse_args()

def prompt_from_3tuple(target, source, target_word):
    return f"If {target} is like {source}, then {target_word} is like"

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

with open(args.file, 'r') as f:
    lines = f.readlines()
        
lines = lines[1:]
lines = [line.strip().split(',') for line in lines]

n_tokens = 0
for row in lines:
    n_tokens += num_tokens_from_string(prompt_from_3tuple(row[0], row[1], row[2]), args.model)

print(f"Estimated number of input tokens: {n_tokens}")
print(f"Estimated cost of GPT 3.5 Turbo 4K: ${n_tokens * 0.001 + len(lines) * args.out_tokens * 0.002}")