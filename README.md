# lm_metaphors
Using language models for detecting metaphors

### Installation

1. Install ![Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).
2. Create a conda environment with the required packages:
```bash
conda env create -f environment.yml
```

### Usage

1. Activate the conda environment:
```bash
conda activate lm-meta
```
2. Run the script:
```bash
bash query_model.sh    # Simple prompting
# OR
bash query_few_shot.sh # In-context prompting
```