"""
This module is the central authority of this repo for tokenization
of strings.

In general, care should be used when removing tokenizers. Vocabularies are generated asynchronously and stored in .json files. These .json files contain the vocabulary using functions named in this module. 

If these functions change names then the vocabularies would need to be remade. 
"""
import datetime
import json
import logging
import os
import time
from functools import partial

import pandas as pd
import spacy

from chatty.utils.multiprocessing import parmap
from research.daily_dialogue import data

nlp = spacy.load('en')
_SUBJECTS = {'nsubj', 'pobj', 'dobj', 'cobj', 'iobj'}
VOCAB_SPLITTER = '-'


def ngramize(gen, ngrams=[1], sep='_'):
    grams = list(gen)
    for ngram_size in ngrams:
        for i in range(ngram_size, len(grams) + 1):
            yield sep.join(grams[i - ngram_size: i])


def subjects_dependency_pos(doc: spacy.tokens.doc.Doc, sep='_'):
    """
    tokenize ROOT_<text>_ROOTPOS_<pos_text>_POS_<token_pos>_DEP_<token_dep>
    """
    for sent in doc.sents:
        root = sent.root
        for tok in sent:
            if tok.dep_ in _SUBJECTS:
                yield sep.join(('ROOT', root.text, "ROOTPOS", root.pos_, "POS", tok.pos_, "DEP", tok.dep_))
                

def pos_lemma(doc: spacy.tokens.doc.Doc, sep='_'):
    """
    tokenize
    LEMMA_<lemma_text>_POS_<pos_text>
    e.g.
    tokenize_pos_obj("He is walking") ->
    ['LEMMA_he_POS_NOUN', 'LEMMA_be_POS_VERB', 'LEMMA_walk_POS_VERB']
    """
    for tok in doc:
        yield sep.join(("LEMMA", tok.lemma_, "POS", tok.pos_))


def pos_and_words(doc: spacy.tokens.doc.Doc, sep='_'):
    for tok in doc:
        yield sep.join(('WORD', tok.text.lower(), 'POS', tok.pos_))

            
def pos(doc: spacy.tokens.doc.Doc, sep='_'):
    for tok in doc:
        yield sep.join(('POS', tok.pos_))


def word(doc: spacy.tokens.doc.Doc, sep='_', lower=True):
    for tok in doc:
        yield sep.join(('WORD', tok.text.lower())) if lower else tok.text


def tokenize(string: str, tokenizers=[], sep='_'):
    "all tokenizers come through here"
    doc = nlp(string)
    for tokenizer in tokenizers:
        for token in tokenizer(doc, sep=sep):
            yield token


def tokenize_as_list(*args, **kwargs):
    "Wrapper for tokenize to return list instead of generator"
    tokens = []
    for tok in tokenize(*args, **kwargs):
        tokens.append(tok)
    return tokens


def _vocab_dir():
    base_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.dirname(os.path.dirname(base_path))
    base_path = os.path.join(base_path, "research")
    base_path = os.path.join(base_path, "daily_dialogue", 'data', 'vocab')
    return base_path


def _vocab_path(*token_types):
    # this ensures that we don't have the same two things
    # with different file names
    token_types = sorted(token_types)
    fpath = VOCAB_SPLITTER.join(["{}".format(tok) for tok in token_types])
    fpath = fpath + ".json"
    base_path = _vocab_dir()
    fpath = os.path.join(base_path, fpath)
    return fpath


def save_vocab(vocab: list, *token_types, **kwargs):
    fpath = _vocab_path(*token_types)
    with open(fpath, 'w') as f:
        output = {
            "vocab": vocab,
            "token_types": token_types
        }
        output.update(kwargs)
        json.dump(output, f, ensure_ascii=False)


def load_vocab(*token_types):
    fpath = _vocab_path(*token_types)
    with open(fpath, 'r') as f:
        return json.load(f)


def make_vocabulary(docs, tokenizers=[], chunksize=25, n_jobs=1):
    """Tokenizes docs and saves to a .json file with a name based on tokenizers used

    Parameters
    ----------
    docs : list or generator
        List of docs that you want to tokenize
    tokenizers : list of 2-tuples of (str, <function>)
        List of tokenizers and a 'name'
    chunksize : int
        Size of chunk of documents to tokenize at a time if running in parallel
    n_jobs : int
        Number of processes to use to make the vocabulary

    Returns
    -------
    None

    Example
    -------
    >>> make_vocabulary(docs, tokenizers=[(ngram, 'ngram'),
                        chunksize=250, n_jobs=-1])
    """
    # names will be used in the savename
    tokenizer_names, tokenizers = zip(*tokenizers)
    # make tokenizer
    tokenizer = partial(tokenize_as_list,
                                  sep='_', tokenizers=tokenizers)
    start = time.time()
    if n_jobs == -1:
        vocab = parmap(tokenizer, docs, chunksize=chunksize)
    elif n_jobs > 0:
        vocab = parmap(tokenizer, docs, chunksize=chunksize, processes=n_jobs)
    else:
        vocab = list(map(tokenizer, docs))
    total_time = time.time() - start

    # save
    save_vocab(vocab, *tokenizer_names, time=total_time)


def list_vocabs():
    dirs = os.listdir(_vocab_dir())
    vocabs = list(filter(lambda x: '.json' in x, dirs))
    vocabs = [i.replace('.json', '') for i in vocabs]
    separated_tokens = tuple([sorted(i.split(VOCAB_SPLITTER)) for i in vocabs])
    return separated_tokens


def word_ngram_2(doc: spacy.tokens.doc.Doc, sep='-'):
    for tok in ngramize(word(doc)):
        yield tok


def first_shot(use_cached_utterances=True, include_test_vocab=False):
    if use_cached_utterances:
        print("### USING CACHED UTTERANCES ###")
    else:
        print("### NOT USING CACHED UTTERANCES ###")
    # SHOULD USE THE TRAIN DATA INSTEAD OF DIALOGUES DIRECTLY SINCE UTTERANCES ARE ALREADY SPLIT
    ### \/ methods below don't correctly separate on '__eou__'
    # import research.daily_dialogue.data as data
    # dialogues = data.dialogues()
    train, _, test, _ = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    
    
    tokenizers = [
        ('subjects_dependency_pos', subjects_dependency_pos),
        ('word_ngram_2', word_ngram_2)
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, chunksize=100, n_jobs=4)
    if include_test_vocab:
        print("ALSO MAKING TEST VOCAB")
        utterances_test = test['utter'].tolist()
        make_vocabulary(utterances_test, tokenizers=tokenizers, chunksize=100, n_jobs=4)

