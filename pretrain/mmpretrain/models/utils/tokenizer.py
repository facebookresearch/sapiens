# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import os

from mmengine.fileio import list_from_file
from transformers import (AutoTokenizer, BartTokenizer, BasicTokenizer,
                          BertTokenizer, BertTokenizerFast, LlamaTokenizer,
                          WordpieceTokenizer)

from mmpretrain.registry import TOKENIZER
from .huggingface import register_hf_tokenizer

register_hf_tokenizer(AutoTokenizer)
register_hf_tokenizer(LlamaTokenizer)


@register_hf_tokenizer()
class BlipTokenizer(BertTokenizerFast):
    """"BlipTokenizer inherit BertTokenizerFast (fast, Rust-based)."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *init_inputs,
        **kwargs,
    ):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            **kwargs,
        )
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        return tokenizer


@register_hf_tokenizer()
class Blip2Tokenizer(BertTokenizer):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *init_inputs,
        **kwargs,
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            **kwargs,
        )
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        return tokenizer


@register_hf_tokenizer()
class OFATokenizer(BartTokenizer):

    vocab_files_names = {
        'vocab_file': 'vocab.json',
        'merges_file': 'merges.txt'
    }

    pretrained_vocab_files_map = {
        'vocab_file': {
            'OFA-Sys/OFA-tiny':
            'https://huggingface.co/OFA-Sys/OFA-tiny/blob/main/vocab.json',
            'OFA-Sys/OFA-medium':
            'https://huggingface.co/OFA-Sys/OFA-medium/blob/main/vocab.json',
            'OFA-Sys/OFA-base':
            'https://huggingface.co/OFA-Sys/OFA-base/blob/main/vocab.json',
            'OFA-Sys/OFA-large':
            'https://huggingface.co/OFA-Sys/OFA-large/blob/main/vocab.json',
        },
        'merges_file': {
            'OFA-Sys/OFA-tiny':
            'https://huggingface.co/OFA-Sys/OFA-tiny/blob/main/merges.txt',
            'OFA-Sys/OFA-medium':
            'https://huggingface.co/OFA-Sys/OFA-medium/blob/main/merges.txt',
            'OFA-Sys/OFA-base':
            'https://huggingface.co/OFA-Sys/OFA-base/blob/main/merges.txt',
            'OFA-Sys/OFA-large':
            'https://huggingface.co/OFA-Sys/OFA-large/blob/main/merges.txt',
        },
    }

    max_model_input_sizes = {
        'OFA-Sys/OFA-tiny': 1024,
        'OFA-Sys/OFA-medium': 1024,
        'OFA-Sys/OFA-base': 1024,
        'OFA-Sys/OFA-large': 1024,
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *init_inputs,
        **kwargs,
    ):
        num_bins = kwargs.pop('num_bins', 1000)
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            **kwargs,
        )
        length = len(tokenizer)
        tokenizer.add_tokens(['<code_{}>'.format(i) for i in range(8192)])
        tokenizer.code_offset = length
        tokenizer.add_tokens(['<bin_{}>'.format(i) for i in range(num_bins)])
        tokenizer.bin_offset = length + 8192
        tokenizer.num_bins = num_bins
        return tokenizer


@TOKENIZER.register_module()
class FullTokenizer(BertTokenizer):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token='[UNK]', max_input_chars_per_word=200)

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        vocab_list = list_from_file(vocab_file)
        for token in vocab_list:
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
        return vocab

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_by_vocab(self, vocab, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)

    @staticmethod
    def convert_tokens_to_string(tokens, clean_up_tokenization_spaces=True):
        """Converts a sequence of tokens (string) in a single string."""

        def clean_up_tokenization(out_string):
            """Clean up a list of simple English tokenization artifacts like
            spaces before punctuations and abbreviated forms."""
            out_string = (
                out_string.replace(' .', '.').replace(' ?', '?').replace(
                    ' !', '!').replace(' ,', ',').replace(" ' ", "'").replace(
                        " n't", "n't").replace(" 'm", "'m").replace(
                            " 's", "'s").replace(" 've",
                                                 "'ve").replace(" 're", "'re"))
            return out_string

        text = ' '.join(tokens).replace(' ##', '').strip()
        if clean_up_tokenization_spaces:
            clean_text = clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def vocab_size(self):
        return len(self.vocab)
