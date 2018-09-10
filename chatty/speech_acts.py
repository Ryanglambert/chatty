from functools import partial
import os
import numpy as np

from chatty import model
from chatty.utils import tokens

# this is related to this bug when using XGBoost
# https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
_tokenizer = partial(tokens.tokenize_as_list,
                     tokenizers=[tokens.chunk_pos_bigram,
                                 tokens.sentence_subj_verb_obj,
                                 tokens.lemma])
clf = model.speech_act_classifier()


def _tokenize_utterances(utter_t_minus_1, utter):
    current = tokens.token_joiner(_tokenizer(utter))
    prev = tokens.token_joiner(_tokenizer(utter_t_minus_1))
    return np.array([prev, current]).reshape(1, -1)


def _get_lags(utterances, lag=1):
    lagged_utterances = [None for i in range(lag)]
    for i in range(len(utterances) - lag):
        lagged_utterances.append(utterances[i])
    return lagged_utterances


def _append_lags(utterances: list) -> list:
    lagged_utterances = _get_lags(utterances, lag=1)
    prev_current_utterances = filter(lambda x: x[0] is not None, zip(lagged_utterances,
                                                                     utterances))
    parsed = []
    for prev, cur in prev_current_utterances:
        parsed.append((prev, cur))
    return parsed

def _marshal_conf(act, conf):
    return {
        'speech_act': act,
        'conf': conf
    }

def parse(utterances: list) -> dict:
    utters_with_lags = _append_lags(utterances)
    tokenized = [_tokenize_utterances(*i) for i in utters_with_lags]
    speech_acts = ["None"] + [clf.predict(toks)[0] for toks in tokenized]
    # first utterance doesn't get classified so these values are empty
    confs = [dict(zip(clf.classes_, [0] * clf.classes_.shape[0]))]
    # The rest of the utterances do get classified
    confs += [dict(zip(clf.classes_, clf.predict_proba(toks)[0])) for toks in tokenized]
    # confs = [_marshal_conf(i, j) for i, j in confs]

    return {
        'speech_acts': speech_acts,
        'conf_speech_acts': confs
    }
