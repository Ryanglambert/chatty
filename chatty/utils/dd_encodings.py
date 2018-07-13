TOPICS = {'1': 'ordinary_life', '2': 'school_life',
          '3': 'culture_education', '4': 'attitude_emotion',
          '5': 'relationship', '6': 'tourism', '7': 'health',
          '8': 'work', '9': 'politics', '10': 'finance'}


ACTS = {'1': 'inform', '2': 'question', '3': 'directive', '4': 'commissive'}


EMOS = {'0': 'no_emotion', '1': 'anger', '2': 'disgust',
        '3': 'fear', '4': 'happiness', '5': 'sadness', '6': 'surprise'}


def _decode(tag, table: dict):
    tag = tag.strip('\n')
    stuff = table.get(tag)
    if not stuff:
        raise Exception(tag)
    else:
        return stuff


def decode_topic(tag):
    return _decode(tag, TOPICS)


def decode_act(act):
    return _decode(act, ACTS)


def decode_emo(emo):
    return _decode(emo, EMOS)
