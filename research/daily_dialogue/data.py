import numpy as np
import os
import pandas as pd
import textblob

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.model_selection import StratifiedKFold

# from chatty.utils import word2vec


CUR_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data')
DFPATH = os.path.join(DATA_DIR, '{}_data_frame.pkl')
NPPATH = os.path.join(DATA_DIR, '{}_matrix.npy')

TOPICS = {'1': 'ordinary_life', '2': 'school_life',
          '3': 'culture_education', '4': 'attitude_emotion',
          '5': 'relationship', '6': 'tourism', '7': 'health',
          '8': 'work', '9': 'politics', '10': 'finance'}
ACTS = {'1': 'inform', '2': 'question', '3': 'directive', '4': 'commissive'}
EMOS = {'0': 'no_emotion', '1': 'anger', '2': 'disgust',
        '3': 'fear', '4': 'happiness', '5': 'sadness', '6': 'surprise'}

np.random.seed(42)


def _topic_stream():
    topics_file = os.path.join(DATA_DIR, 'dialogues_topic.txt')
    return open(topics_file, 'rb').readlines()


def _decode_topic(tag):
    tag = tag.strip('\n')
    stuff = TOPICS.get(tag)
    if not stuff:
        raise Exception(tag)
    else:
        return stuff


def _decode_act(act):
    return ACTS.get(act)


def _decode_emo(emo):
    return EMOS.get(emo)


def _file_streams():
    file_paths = ['dialogues_text.txt',
                  'dialogues_act.txt',
                  'dialogues_emotion.txt']
    file_paths = [os.path.join(DATA_DIR, i)
                  for i in file_paths]
    return [open(i, 'rb').readlines()
            for i in file_paths]


def _dialogues_path():
    return os.path.join(DATA_DIR, 'dialogues_text.txt')


def dialogues():
    fpath = _dialogues_path()
    with open(fpath, 'r') as stream:
        return stream.readlines()


def _to_unicode(string):
    return str(string, 'utf-8')


def _check_short(string):
    if not len(string) > 0:
        return None
    return string


def _sentiment(string):
    blob = textblob.TextBlob(string)
    polarity, subjectivity = blob.sentiment
    return polarity, subjectivity


def _parse_utterances(dial, act, emo):
    utters = dial.strip('\n').split('__eou__')
    acts = act.strip('\n').split(' ')
    emos = emo.strip('\n').split(' ')

    # conversation = []
    first_speaker = True
    for utter, act, emo in zip(utters, acts, emos):
        speaker = 'person_a'
        if not first_speaker:
            speaker = 'person_b'
        first_speaker = not first_speaker
        utter = _check_short(utter)
        if utter:
            polarity, subjectivity = _sentiment(utter)
            yield (speaker, utter, _decode_act(act), _decode_emo(emo), polarity, subjectivity)


def _get_convs():
    dials, acts, emos = _file_streams()
    topics = _topic_stream()
    for conv_id, (dial, act, emo, topic) in enumerate(zip(dials, acts, emos, topics)):
        dial, act, emo, topic = [_to_unicode(i) for i in (dial, act, emo, topic)]
        topic = _decode_topic(topic)
        for utterance in _parse_utterances(dial, act, emo):
            yield utterance + (conv_id, topic)


def get_biggest_drawdown(s):
    "returns the largest net decrease in sentiment"
    i = np.argmax(np.maximum.accumulate(s) - s)
    j = np.argmax(s[:i])
    return s[i] - s[j]


def get_biggest_drawup(s):
    "returns the largest net increase in sentiment"
    i = np.argmin(np.minimum.accumulate(s) - s)
    j = np.argmin(s[:i])
    return s[i] - s[j]


def get_lags(df, lag_range=[1, 2]):
    "Gets lags values within conversation"
    groupby_conv = df.groupby(df.index.get_level_values(0))
    train_lags = pd.concat([groupby_conv.shift(i) 
                            for i in lag_range],
                           axis=1)
    new_cols = []
    for col in df.columns:
        for lag_num in lag_range:
            new_cols.append(col + "_t-" + str(lag_num))
    train_lags.columns = new_cols
    return train_lags


def _make_df(convs):
    df = pd.DataFrame(convs,
                      columns=['person',
                               'utter',
                               'act',
                               'emo',
                               'polarity',
                               'subjectivity',
                               'conv',
                               'topic'])
    df.set_index(['conv'], inplace=True)
    df.set_index([df.index, df.groupby(df.index).cumcount()], inplace=True)
    df['first_utterance'] = df.index.get_level_values(1)\
        .map(lambda x: 1 if x == 0 else 0)

    return df


def _data():
    return _make_df(_get_convs())


def _make_train_test_split(df, test_size=1000):
    conv_ids = df.index.get_level_values(0).unique()
    test_indices = np.random.choice(conv_ids, size=test_size, replace=False)
    train_indices = np.setdiff1d(conv_ids, test_indices)
    return df.loc[train_indices], df.loc[test_indices]


def _make_pickles_pd(df, name):
    df.to_pickle(DFPATH.format(name))


def _read_pickle_pd(name):
    return pd.read_pickle(DFPATH.format(name))


def _make_pickles_np(arr, name):
    np.save(NPPATH.format(name), arr)


def _read_pickle_np(name):
    return np.load(NPPATH.format(name))


def get_data(test_size=1000, use_cached=False):
    if use_cached:
        print('Using Cached')
        train = _read_pickle_pd('train')
        test = _read_pickle_pd('test')
    else:
        print('Not Using Cached')
        print('Parsing Labeled Conversations')
        data = _data()
        train, test = _make_train_test_split(data)
        _make_pickles_pd(train, 'train')
        _make_pickles_pd(test, 'test')
    return train, test


def cv_stratified_shuffle(X: np.array,
                          y: np.array,
                          model,
                          splits=5,
                          probability=True):
    """Rusn stratified shuffle split on X, y, with given model, for n splits

    Parameters
    ----------
    X : np.array
    y : np.array
    model : sklearn.base.BaseEstimator
    splits : int
        number of folds for cross validation
    upsample : str or None
        Can specify either 'ADASYN' or 'SMOTE' or 'random'
        If 'random' is specified, then the features aren't used
        (you'll probably do this for text)

    Returns
    -------
    results : dict
        e.g.
        {'y_true': [np.array, ... ],
         'y_proba': [np.array, ... ],
         'models': [sklearn.model, ... ],
         'classes': ['directive', 'commissive' ... ]}
    """
    y_true = []
    y_proba = []
    models = []
    sss = StratifiedKFold(n_splits=splits, shuffle=True)
    for train_index, val_index in sss.split(X, y):
        print('Training')
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(x_train, y_train)
        if probability:
            proba = model.predict_proba(x_val)
        else:
            proba = model.predict(x_val)
        y_true.append(y_val)
        y_proba.append(proba)
        models.append(model)
    classes = model.classes_
    return {
        'y_true': y_true,
        'y_proba': y_proba,
        'models': models,
        'classes': classes
    }


def results_to_df(results, X=None, columns=None):
    y_true = np.concatenate(results['y_true']).reshape(-1, 1)
    y_proba = np.concatenate(results['y_proba'])
    classes = results['classes']
    df_proba = pd.DataFrame(y_proba, columns=classes)
    df_true = pd.DataFrame(y_true, columns=['y_true'])
    df = pd.concat([df_true, df_proba], axis=1)
    return df
