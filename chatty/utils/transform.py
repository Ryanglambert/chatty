def parse_conversation(conv):
    # TODO:
    # Find the appropriate utterance splitter
    # This will likely be accomplished by finding a repeating pattern
    # split the conversation on this pattern and tag each utterance as belonging to person a or person b
    raise NotImplementedError


# something like this
def _parse_utterance(utter):
    return {
        'utterance': 'Stuff',
        'polarity': 0.,
        'subjectivity': 0.,
        'sentiment': 'happy',
        'dialogue_act': 'inform'
    }