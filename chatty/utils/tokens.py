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

from chatty.utils.methodmultiprocessing import parmap
from research.daily_dialogue import data

nlp = spacy.load('en')
_SUBJECTS = {'nsubj', 'pobj', 'dobj', 'cobj', 'iobj'}
VOCAB_TITLE_SPLITTER = '-'
TOKEN_SPLITTER = '*^'


def token_splitter(toks: str):
    return toks.split(TOKEN_SPLITTER)


def token_joiner(toks: list):
    return TOKEN_SPLITTER.join(toks)


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


def lemma(doc: spacy.tokens.doc.Doc, sep='_'):
    for tok in doc:
        yield sep.join(("LEMMA", tok.lemma_))


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


def tokenize_pipe(doc: spacy.tokens.doc.Doc, tokenizers=[], sep='-'):
    tokens = []
    for tokenizer in tokenizers:
        for token in tokenizer(doc, sep=sep):
            tokens.append(token)
    return tokens


def tokenize_as_list_pipe(*args, **kwargs):
    tokens = []
    for tok in tokenize_pipe(*args, **kwargs):
        tokens.append(tok)
    return tokens


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
    fpath = VOCAB_TITLE_SPLITTER.join(["{}".format(tok) for tok in token_types])
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


def make_vocabulary(docs, tokenizers=[], chunksize=25, n_jobs=1, verbose=False):
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
    start = time.time()
    if n_jobs > 0:
        tokenizer = partial(tokenize_as_list_pipe,
                                      sep='_', tokenizers=tokenizers)
        # vocab = parmap(tokenizer, docs, chunksize=chunksize, processes=n_jobs, verbose=verbose)
        vocab = [tokenizer(i) for i in nlp.pipe(docs, batch_size=chunksize, n_threads=n_jobs)]
    else:
        tokenizer = partial(tokenize_as_list,
                                      sep='_', tokenizers=tokenizers)
        vocab = list(map(tokenizer, docs))
    total_time = time.time() - start

    # save
    try:
        save_vocab(vocab, *tokenizer_names, time=total_time)
    except:
        print(vocab)


def list_vocabs():
    dirs = os.listdir(_vocab_dir())
    vocabs = list(filter(lambda x: '.json' in x, dirs))
    vocabs = [i.replace('.json', '') for i in vocabs]
    separated_tokens = tuple([sorted(i.split(VOCAB_TITLE_SPLITTER)) for i in vocabs])
    return separated_tokens


def word_ngram_2(doc: spacy.tokens.doc.Doc, sep='-'):
    for tok in ngramize(word(doc), ngrams=[2]):
        yield tok


def lemma_ngram_2(doc: spacy.tokens.doc.Doc, sep='-'):
    for tok in ngramize(lemma(doc), ngrams=[2]):
        yield tok


def lemma_ngram_3(doc: spacy.tokens.doc.Doc, sep='-'):
    for tok in ngramize(lemma(doc), ngrams=[3]):
        yield tok


def pos_ngram_2(doc: spacy.tokens.doc.Doc, sep='-'):
    for tok in ngramize(pos(doc), ngrams=[2]):
        yield tok


def pos_ngram_3(doc: spacy.tokens.doc.Doc, sep='-'):
    for tok in ngramize(pos(doc), ngrams=[3]):
        yield tok

    
def dependencies(doc: spacy.tokens.doc.Doc, sep='-'):
    for tok in doc:
        lem = tok.pos_ if not tok.is_title else tok.text
        head_lem = tok.head.lemma_.lower() if not tok.head.is_title else tok.head.text
        yield "{0}/{1} <--{2}-- {3}/{4}".format(
            tok.pos_, tok.tag_, tok.dep_,
            tok.head.pos_, tok.head.tag_
            )


def subj_verb_obj(sent: spacy.tokens.span.Span, sep='-'):
    root = "ROOT({})".format(sent.root.head.text.lower())
    verbs = []
    subjects = []
    objects = []
    intjs = []
    auxs = []
    for child in sent.root.head.children:
        if 'subj' in child.dep_:
            subjects.append('SUBJ({})'.format(child.pos_))
        elif 'obj' in child.dep_:
            objects.append('OBJ({})'.format(child.pos_))
        elif 'intj' in child.dep_:
            intjs.append("INTJ({})".format(child.lemma_))
        elif 'aux' in child.dep_:
            auxs.append("AUX({})".format(child.lemma_))
        elif 'VERB' in child.pos_:
            verbs.append("VERB({})".format(child.lemma_))

    objects = list(set(objects))
    subjects = list(set(subjects))
    intjs = list(set(intjs))
    auxs = list(set(auxs))
    tokens = subjects + [root] + objects + verbs + intjs + auxs
    return sep.join(tokens)


def sentence_subj_verb_obj(doc: spacy.tokens.doc.Doc, sep='-'):
    for sent in doc.sents:
        tok = subj_verb_obj(sent, sep=sep)
        yield tok


def chunk_pos_bigram(doc: spacy.tokens.doc.Doc, sep='-'):
    for chunk in doc.noun_chunks:
        yield sep.join(('CHK_ROOT_POS', chunk.root.pos_,
                        'CHK_ROOT_HEAD', chunk.root.head.pos_))


def chunk_pos_two_bigram(doc: spacy.tokens.doc.Doc, sep='-'):
    for tok in ngramize(chunk_pos_bigram(doc), ngrams=[2]):
        yield tok


def ninth_shot(use_cached_utterances=True, chunksize=100, n_jobs=1, verbose=False):
    train, test = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    tokenizers = [
        # ('chunk_pos_two_bigram', chunk_pos_two_bigram),
        ('chunk_pos_bigram', chunk_pos_bigram),
        ('word', lemma),
        # ('lemma_ngram_2', lemma_ngram_2),
        ('sentence_subj_verb_obj', sentence_subj_verb_obj),
        # ('pos_ngram_2', pos_ngram_2),
        # ('lemma_ngram_3', lemma_ngram_3),
        # ('pos_ngram_3', pos_ngram_3),
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


def eighth_shot(use_cached_utterances=True, chunksize=100, n_jobs=1, verbose=False):
    train, _, test, _ = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    tokenizers = [
        # ('chunk_pos_two_bigram', chunk_pos_two_bigram),
        # ('chunk_pos_bigram', chunk_pos_bigram),
        # ('word', lemma),
        # ('lemma_ngram_2', lemma_ngram_2),
        # ('pos_ngram_2', pos_ngram_2),
        # ('lemma_ngram_3', lemma_ngram_3),
        ('pos_ngram_3', pos_ngram_3),
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


def seventh_shot(use_cached_utterances=True, chunksize=100, n_jobs=1, verbose=False):
    train, _, test, _ = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    tokenizers = [
        ('chunk_pos_two_bigram', chunk_pos_two_bigram),
        ('chunk_pos_bigram', chunk_pos_bigram),
        ('word', lemma),
        ('lemma_ngram_2', lemma_ngram_2),
        ('pos_ngram_2', pos_ngram_2),
        # ('lemma_ngram_3', lemma_ngram_3),
        ('pos_ngram_3', pos_ngram_3),
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


def sixth_shot(use_cached_utterances=True, chunksize=100, n_jobs=1, verbose=False):
    train, _, test, _ = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    tokenizers = [
        ('chunk_pos_bigram', chunk_pos_bigram),
        ('word', lemma),
        ('lemma_ngram_2', lemma_ngram_2),
        ('pos_ngram_2', pos_ngram_2),
        ('lemma_ngram_3', lemma_ngram_3),
        ('pos_ngram_3', pos_ngram_3),
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


def fifth_shot(use_cached_utterances=True, chunksize=100, n_jobs=1, verbose=False):
    train, _, test, _ = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    tokenizers = [
        ('chunk_pos_bigram', chunk_pos_bigram),
        ('word', lemma),
        ('lemma_ngram_2', lemma_ngram_2),
        ('pos_ngram_2', pos_ngram_2)
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


def fourth_shot(use_cached_utterances=True, chunksize=100, n_jobs=1, verbose=False):
    train, _, test, _ = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    tokenizers = [
        ('chunk_pos_bigram', chunk_pos_bigram),
        ('word', lemma),
        ('pos_ngram_2', lemma_ngram_2)
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


def first_shot(use_cached_utterances=True, chunksize=100, n_jobs=1, verbose=False):
    train, _, test, _ = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    tokenizers = [
        ('subjects_dependency_pos', subjects_dependency_pos),
        ('word_ngram_2', word_ngram_2)
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


def second_shot(use_cached_utterances=True, chunksize=100, n_jobs=1, verbose=False):
    train, _, test, _ = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    tokenizers = [
        ('lemma_ngram_2', lemma_ngram_2),
        ('lemma_', lemma),
        ('pos_ngram_2', pos_ngram_2),
        ('pos', pos)
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)


def third_shot(use_cached_utterances=True, chunksize=100, n_jobs=1, verbose=False):
    train, _, test, _ = data.get_data(use_cached=use_cached_utterances)
    utterances = train['utter'].tolist()
    tokenizers = [
        ('lemma_ngram_2', lemma_ngram_2),
        ('lemma_', lemma),
        ('pos_ngram_2', pos_ngram_2),
        ('pos', pos),
        ('dependencies', dependencies)
    ]
    make_vocabulary(utterances, tokenizers=tokenizers, n_jobs=n_jobs, chunksize=chunksize, verbose=verbose)

if __name__ == '__main__':
    # first_shot(n_jobs=36, chunksize=1000)
    # second_shot(n_jobs=4, chunksize=1000, verbose=True)
    # third_shot(n_jobs=4, chunksize=1000, verbose=True)
    # fourth_shot(n_jobs=4, chunksize=1000, verbose=True)
    # fifth_shot(n_jobs=4, chunksize=1000, verbose=True)
    # sixth_shot(n_jobs=20, chunksize=1000, verbose=True)
    # seventh_shot(n_jobs=30, chunksize=1000, verbose=True)
    # eighth_shot(n_jobs=30, chunksize=1000, verbose=True)
    ninth_shot(n_jobs=30, chunksize=1000, verbose=True)

