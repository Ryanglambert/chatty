import pytest


from chatty.extractors import conversation_to_utterances


def test_split_extractor():
    # copied and pasted the first utterance to the second assertion
    conversation = (" Jinnd319 [2:21 PM]"
    "I need some help with understanding something painfully basic. In pandas, when you compare a series made like this,  `data_frame ['column']` using something like `.le` what actually happens? Does it use something indexes to make sure the right things get compared or does it just compare things based on order?"

    "evamicur [2:50 PM]"
    "I'm not sure I understand @Jinnd319 . The index operation should create (or get a reference to) a pd.Series then do the comparison, which would behave just like any series operation would"

    "Jinnd319 [2:55 PM]"
    "@evamicur so in my example `column` would be compared without any consideration for values in the rest of the dataframe?")

    utterances = conversation_to_utterances(conversation)
    assert len(utterances) == 3
    assert utterances[0] == \
        "I need some help with understanding something painfully basic. In pandas, when you compare a series made like this,  `data_frame ['column']` using something like `.le` what actually happens? Does it use something indexes to make sure the right things get compared or does it just compare things based on order?"
