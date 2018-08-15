import pytest


def test_analyze(client):
    response = client.post(
        '/analyze',
        content_type="application/json",
        json={'text': 'test __eou__ another'}
    )
    payload = response.get_json()
    for field, value_type in [
        ('conf_next_sentiment', float),
        ('conf_sentiment', list),
        ('next_sentiment', str),
        ('sentiment', list),
        ('speech_acts', list),
        ('utterances', list)
    ]:
        assert field in payload
        assert isinstance(payload[field], value_type)
