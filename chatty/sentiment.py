import numpy as np


SENTIMENTS = [
    'anger',
    'disgust',
    'sadness',
    'happiness',
    'fear',
    'surprise'
]


def parse(utterances: list) -> dict:
    sents = [np.random.choice(SENTIMENTS)
             for i in utterances]
    conf_sents = [np.random.rand()
                  for i in utterances]
    return {
        'sentiment': sents,
        'conf_sentiment': conf_sents
    }


def parse_next(tagged_utterances):
    return {
        'next_sentiment': np.random.choice(SENTIMENTS),
        'conf_next_sentiment': np.random.random()
    }