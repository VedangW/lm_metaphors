import csv
import tqdm
from collections import defaultdict

domains = defaultdict(list)

def generate_in_context_prompt(target, data):
    for i in range(len(data)):
        prompt = ""
        for j in range(len(data)):
            if i != j:
                source, target_word, source_word, _ = data[j]
                prompt += f"If '{target}' is like '{source}', then '{target_word}' is like '{source_word}'. "

        prompt += f"If '{target}' is like '{data[i][0]}', then '{data[i][1]}' is like what? Answer in one word."

        yield prompt, data[i][2], data[i][3]

with open('data/scan.csv', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='/')
        for i, row in enumerate(tqdm(reader)):
            if i == 0:
                continue
            
            target, source, target_word, source_word = row[:4]
            alternatives = row[4:-1]
            alternatives = [x.strip().replace('"', '') for x in alternatives if x.strip() != '']
            
            domains[target].append((source, target_word, source_word, alternatives))

print(domains.keys())