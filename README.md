# Japanese Phonemes Separation

## Description

A tool for segmenting Japanese phoneme sequences according to word boundaries using an EM-based alignment model.

## Installation

```bash
# Clone repository
git clone https://github.com/Kiri4s/Japanese-phonemes-separation.git
cd Japanese-phonemes-separation

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

## Usage

```bash
uv run main.py
```

This will:
- Train the alignment model on training data
- Evaluate performance on test set
- Print detailed comparison of predicted and true segmentations

## Dataset Format

Input data should be CSV files with the following columns:

- `id`: Example ID
- `phrase`: Japanese text 
- `split_phrase`: Japanese text with word boundaries marked by spaces
- `ipa`: Raw IPA phoneme sequence
- `split_ipa`: (optional) IPA sequence with word boundaries marked by `<space>` tokens

## Method

1. Japanese text is tokenized into moras
2. EM algorithm learns mapping probabilities from moras to phoneme subsequences
3. Viterbi algorithm finds optimal alignment between moras and phonemes
4. Word boundaries from kana text are transferred to phoneme sequence
