import numpy as np

SENTIMENTS = [
    'anger',
    'disgust',
    'sadness',
    'happiness',
    'fear',
    'surprise'
]
ACTS = [
    'commissive',
    'directive',
    'inform',
    'question'
]


def analyze(text):
    utterances = text.split('__eou__')
    speech_acts = [np.random.choice(ACTS)
                   for i in utterances]
    conf_speech_acts = [np.random.rand()
                        for i in utterances]
    sents = [np.random.choice(SENTIMENTS)
             for i in utterances]
    conf_sents = [np.random.rand()
                  for i in utterances]
    next_sentiment = np.random.choice(SENTIMENTS)
    conf_next_sentiment = np.random.random()
    return {
        'utterances': utterances,
        'sentiment': sents,
        'conf_sentiment': conf_sents,
        'speech_acts': speech_acts,
        'conf_speech_acts': conf_speech_acts,
        'next_sentiment': next_sentiment,
        'conf_next_sentiment': conf_next_sentiment
    }
