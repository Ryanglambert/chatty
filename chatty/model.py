from sklearn.externals import joblib
import os


def get_model_path(classifier='speech_act'):
    base_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.dirname(os.path.dirname(base_path))
    base_path = os.path.join(base_path, "chatty")
    base_path = os.path.join(base_path, "research")
    base_path = os.path.join(base_path, "daily_dialogue")
    return os.path.join(base_path, classifier + '.pkl')


def save_model(clf, classifer: str):
    model_path = get_model_path(classifier)
    joblib.dump(clf, model_path)


def load_model(classifier='speech_act'):
    model_path = get_model_path(classifier)
    return joblib.load(model_path)


def speech_act_classifier():
    return load_model('speech_act')

