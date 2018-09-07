from functools import partial
import numpy as np

from chatty import model
from chatty.utils import tokens


ACTS = [
    'commissive',
    'directive',
    'inform',
    'question'
]


_tokenizer = partial(tokens.tokenize_as_list, tokenizers=[tokens.chunk_pos_bigram,
                                                          tokens.sentence_subj_verb_obj,
                                                          tokens.lemma])
clf = model.speech_act_classifier()


def tokenize_utterances(utter_t_minus_1, utter):
    current = tokens.token_joiner(_tokenizer(utter))
    prev = tokens.token_joiner(_tokenizer(utter_t_minus_1))
    return np.array([prev, current]).reshape(1, -1)


def get_lags(utterances, lag=1):
    lagged_utterances = [None for i in range(lag)]
    for i in range(len(utterances) - lag):
        lagged_utterances.append(utterances[i])
    return lagged_utterances


def append_lags(utterances: list) -> list:
    lagged_utterances = get_lags(utterances, lag=1)
    prev_current_utterances = filter(lambda x: x[0] is not None, zip(lagged_utterances,
                                                                     utterances))
    parsed = []
    for prev, cur in prev_current_utterances:
        parsed.append((prev, cur))
    return parsed
#     for prev, cur in zip(lagged_utterances, utterances):

def parse(utterances: list) -> dict:
    utters_with_lags = append_lags(utterances)
    tokenized = [tokenize_utterances(*i) for i in utters_with_lags]
    speech_acts = ["None"] + [clf.predict(toks)[0] for toks in tokenized]
    # first utterance doesn't get classified
    confs = ["N/A", list(zip(clf.classes_, [0] * clf.classes_.shape[0]))]
    confs += [list(zip(clf.classes_, clf.predict_proba(toks)[0])) for toks in tokenized]
    return {
        'speech_acts': speech_acts,
        'conf_speech_acts': confs
    }
