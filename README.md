# Robust Pronoun Use Fidelity with English LLMs: Are they Reasoning, Repeating, or Just Biased?

This repository contains code, data and human annotations for our paper.

## Getting started

Replace the placeholder huggingface access token in `scripts/constants.py` with your own. This is necessary for running Llama-2 models. Refer to the [Llama-2 documentation](https://huggingface.co/meta-llama) for the specifics.

## Code

- `constants.py`: secrets, API keys and such
- `pronouns.py`: parametrized list of pronouns we use in the paper (simply extend this dictionary to evaluate on more pronouns)

## Data

We provide our newly constructed templates in a zipped data file. Please unzip it with the password `vogelbeobachtung131719`, using a command like `unzip -P PASSWORD FILE.zip`. It contains two files:
* `task.tsv`: task templates where each row contains a sentence for a given occupation, participant and pronoun type; the word column is equal to occupation because it is the answer for the coreference
* `context.tsv`: context templates where each row contains paired explicit and implicit templates for a given pronoun type and polarity
