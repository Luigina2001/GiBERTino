import random

import numpy as np
import torch
from transformers import MarianTokenizer, MarianMTModel


def calculate_probability(n: int, p: float):
    """
    Calculate probability based on binomial distribution.

    Args:
        n (int): Number of trials.
        p (float): Probability of success.

    Returns:
        int: Randomly generated binomial value (0 or 1).
    """
    return np.random.binomial(n=n, p=p)


TGT_LANGS = ['fr', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo',
             'es', 'it', 'pt', 'gl', 'lad', 'an',
             'mwl', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la']
MAX_MODEL_LENGTH = 85  # max token length returned by the tokenizer on the longest sentence
TOKENS_RANGE = 3


class BackTranslation:
    """
    Apply back translation to a text with a given probability.
    """

    def __init__(self, src_translator: str = "Helsinki-NLP/opus-mt-en-ROMANCE",
                 tgt_translator: str = "Helsinki-NLP/opus-mt-ROMANCE-en",
                 p: float = 0.5):
        """
        Initialize the BackTranslation module.

        Args:
            src_translator (str): Name of the model for source language translation (back translation).
            tgt_translator (str): Name of the model for target language translation.
            p (float): Probability of applying back translation.
        """
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.p = p
        self.src_translator = MarianMTModel.from_pretrained(src_translator).to(
            self.device)
        self.src_tokenizer = MarianTokenizer.from_pretrained(src_translator)
        self.tgt_translator = MarianMTModel.from_pretrained(tgt_translator).to(
            self.device)
        self.tgt_tokenizer = MarianTokenizer.from_pretrained(tgt_translator)

    def translate(self, sample, back: bool = False):
        """
        Translate a text using the specified translator.

        Args:
            sample (str): Input text.
            back (bool): If True, perform back translation.

        Returns:
            str: Translated text.
        """
        tokens = self.src_tokenizer(sample, return_tensors="pt",
                                    padding=True) if back is False else \
            self.tgt_tokenizer(sample, return_tensors="pt", padding=True)
        tokens = tokens.to(self.device)

        translated = self.src_translator.generate(**tokens,
                                                  max_new_tokens=MAX_MODEL_LENGTH) if back is False else \
            self.tgt_translator.generate(**tokens,
                                         max_new_tokens=MAX_MODEL_LENGTH)

        # if the translated sample contains more tokens than the specified threshold, return the original
        # sample
        translated = translated if len(translated) <= (
                    len(tokens) * TOKENS_RANGE) else tokens
        translated = translated.to("cpu")

        # translated text
        return [self.src_tokenizer.decode(t, skip_special_tokens=True) for t in
                translated][0] \
            if back is not True else \
            [self.tgt_tokenizer.decode(t, skip_special_tokens=True) for t in
             translated][0]

    def __call__(self, sample: str):
        """
        Apply back translation to a text with a certain probability.

        Args:
            sample (str): Input text.

        Returns:
            str: Transformed text.
        """
        bit = calculate_probability(n=1, p=self.p)

        if bit == 1:
            # insert >>2 character language code<< at the beginning of the text to define the target language
            tgt_lang = random.choice(TGT_LANGS)
            sample = f">>{tgt_lang}<< " + sample
            return self.translate(self.translate(sample), back=True)
        else:
            return sample
