import functools
import json
import logging
import os
import spacy
import time

from chatty.utils.multiprocessing import parmap


nlp = spacy.load('en')
subjects = {'nsubj', 'pobj', 'dobj', 'cobj', 'iobj'}


def root_to_leave_dep(tok, last_head=None):
    if tok.head == last_head:
        return tok
    else:
        return root_to_leave_dep(tok.head, last_head=tok.head)


def tokenize_subjects(doc: spacy.tokens.doc.Doc, sep='_'):
    """
    tokenize ROOT_<text>_ROOTPOS_<pos_text>_POS_<token_pos>_DEP_<token_dep>
    """
    for sent in doc.sents:
        root = sent.root
        for tok in sent:
            if tok.dep_ in subjects:
                yield sep.join(('ROOT', root.text, "ROOTPOS", root.pos_, "POS", tok.pos_, "DEP", tok.dep_))
                

def tokenize_pos_obj(doc: spacy.tokens.doc.Doc, sep='_'):
    """
    tokenize
    LEMMA_<lemma_text>_POS_<pos_text>


    e.g.
    tokenize_pos_obj("He is walking") ->
    ['LEMMA_he_POS_NOUN', 'LEMMA_be_POS_VERB', 'LEMMA_walk_POS_VERB']
    """
    for tok in doc:
        yield sep.join(("LEMMA", tok.lemma_, "POS", tok.pos_))


def tokenize_words(doc: spacy.tokens.doc.Doc, sep='_'):
    for tok in doc:
        yield sep.join((tok.lemma_, tok.pos_))


def ngram_pos_and_words(doc: spacy.tokens.doc.Doc, sep='_', ngrams=[1]):
    for ngram_size in ngrams:
        for i in range(0, len(doc) - ngram_size):
            yield sep.join([sep.join((i.text.lower(), i.pos_)) for i in doc[i: i + ngram_size]])
            
            
def ngram_pos(doc: spacy.tokens.doc.Doc, sep='_', ngrams=[1]):
    for ngram_size in ngrams:
        for i in range(0, len(doc) - ngram_size):
            yield sep.join([j.pos_ for j in doc[i: i + ngram_size]])
            
            
def ngram(doc: spacy.tokens.doc.Doc, sep='_', ngrams=[1]):
    for ngram_size in ngrams:
        for i in range(0, len(doc) - ngram_size):
            yield sep.join([j.text.lower() for j in doc[i: i + ngram_size]])


def tokenize(string: str, tokenizers=[], sep='_'):
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


def _vocab_path(*token_types):
    # this ensures that we don't have the same two things
    # with different file names
    token_types = sorted(token_types)
    fpath = "_".join(["{}".format(tok) for tok in token_types])
    fpath = fpath + ".json"
    base_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.dirname(os.path.dirname(base_path))
    base_path = os.path.join(base_path, "research")
    base_path = os.path.join(base_path, "daily_dialogue", 'data', 'vocab')
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
    # setup logging
    logging.basicConfig(
        filename=__name__ + ".log",
        level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)
    logger.info("Building Vocabulary Starting at: {}".format(
        time.clock()
    ))

    # names will be used in the savename
    tokenizer_names, tokenizers = zip(*tokenizers)
    # make tokenizer
    tokenizer = functools.partial(tokenize_as_list,
                                  sep='_', tokenizers=tokenizers)
    start = time.time()
    if n_jobs == -1:
        vocab = parmap(tokenizer, docs, chunksize=chunksize)
    else:
        vocab = list(map(tokenizer, docs))
    total_time = time.time() - start

    # save
    save_vocab(vocab, *tokenizer_names, time=total_time)
