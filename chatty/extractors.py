import re


SPLITTER = re.compile(r"(?:.+\s\[\d{1,3}\:\d{1,3}\s\w{1,2}\])")


def conversation_to_utterances(string: str) -> list:
    utterances = []
    if '__eou__' not in string:
        utterances = _split_slack_conversation(string)
    else:
        utterances = string.split('__eou__')
    return utterances


def _split_slack_conversation(string: str) -> list:
    return SPLITTER.split(string)[1:]
