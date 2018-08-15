import re


SPLITTER = re.compile(r"(?:\b(?:\w|\d)+\s\[\d{1,3}\:\d{1,3}\s\w{1,2}\])")


def conversation_to_utterances(string: str) -> list:
    if '__eou__' not in string:
        return _split_slack_conversation(string)
    return string.split('__eou__')


def _split_slack_conversation(string: str) -> list:
    return SPLITTER.split(string)[1:]
