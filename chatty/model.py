import pickle
import os


def _get_model_path(classifier='speech_act'):
    base_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.dirname(os.path.dirname(base_path))
    base_path = os.path.join(base_path, "chatty")
    base_path = os.path.join(base_path, "research")
    base_path = os.path.join(base_path, "daily_dialogue")
    return os.path.join(base_path, classifier + '.pkl')


def save_model(clf, classifier='speech_act'):
    model_path = _get_model_path(classifier)
    print("saving: {}".format(model_path))
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)


def load_model(classifier='speech_act'):
    model_path = _get_model_path(classifier)
    print("loading: {}".format(model_path))
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def speech_act_classifier():
    return load_model('speech_act')

