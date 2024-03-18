# lm_metaphors
Using language models for detecting metaphors

### Installation

1. Install ![Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).
2. Set up the Transformers cache to avoid downloading the same model multiple times and ensure it has a large amount of space available.
3. Create a conda environment with the required packages.

```bash
export TRANSFORMERS_CACHE=/path/to/large/data/dir/transformers_cache
conda env create -f environment.yml
```

### Usage

Activate the conda environment and run the script.
```bash
conda activate lm-meta

bash query_model.sh    # Simple prompting
# OR
bash query_few_shot.sh # In-context prompting
```