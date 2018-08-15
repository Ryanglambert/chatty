import numpy as np

ACTS = [
    'commissive',
    'directive',
    'inform',
    'question'
]

def parse(utterances: list) -> dict:
    speech_acts = [np.random.choice(ACTS)
                   for i in utterances]
    conf_speech_acts = [np.random.rand()
                        for i in utterances]
    return {
        'speech_acts': speech_acts,
        'conf_speech_acts': conf_speech_acts
    }
