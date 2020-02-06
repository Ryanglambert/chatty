import pytest

from chatty.analyze import analyze_slack


@pytest.mark.parametrize(
    "input_text",
    [
        "How are you?"
        "__eou__"
        "I'm good and you?"
        "__eou__"
        "I'm pretty good thanks for asking",
        " Jinnd319 [2:21 PM]"
        "I need some help with understanding something painfully basic. In pandas, when you compare a series made like this,  `data_frame ['column']` using something like `.le` what actually happens? Does it use something indexes to make sure the right things get compared or does it just compare things based on order?"

        "evamicur [2:50 PM]"
        "I'm not sure I understand @Jinnd319 . The index operation should create (or get a reference to) a pd.Series then do the comparison, which would behave just like any series operation would"

        "Jinnd319 [2:55 PM]"
        "@evamicur so in my example `column` would be compared without any consideration for values in the rest of the dataframe?"
    ]
)
def test_analyze(input_text):
    output = analyze_slack(input_text)
    for field_name, field_type in [
        ('utterances', list),
        ('sentiment', list),
        ('conf_sentiment', list),
        ('speech_acts', list),
        ('conf_speech_acts', list),
        ('next_sentiment', str),
        ('conf_next_sentiment', float),
    ]:
        assert field_name in output
        assert isinstance(output[field_name], field_type)
