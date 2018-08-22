import json
import os
import spacy


nlp = spacy.load('en')
subjects = {'nsubj', 'pobj', 'dobj', 'cobj', 'iobj'}


def tag(prefix="", suffix=""):
    def decorator(func):
        def new(string):
            output = func(string)
            for i in output:
                yield prefix + i + suffix
        return new
    return decorator


def root_to_leave_dep(tok, last_head=None):
    if tok.head == last_head:
        return tok
    else:
        return root_to_leave_dep(tok.head, last_head=tok.head)


def tokenize_subjects(doc: spacy.tokens.doc.Doc, sep='_'):
    for sent in doc.sents:
        root = sent.root
        for tok in sent:
            if tok.dep_ in subjects:
                yield sep.join(('ROOT', root.text, "ROOTPOS", root.pos_, "POS", tok.pos_, "DEP", tok.dep_))
                

def tokenize_pos_obj(doc: spacy.tokens.doc.Doc, sep='_'):
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


def save_vocab(vocab: list, *token_types):
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

    with open(fpath, 'w') as f:
        json.dump({"vocab": vocab,
                   "token_types": token_types},
                   f, ensure_ascii=False)

    return fpath
            