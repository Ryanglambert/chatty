import pytest


def test_endpoint(client):
    response = client.post(
        '/analyze_slack',
        content_type="application/json",
        json={'text': 'test __eou__ another'}
    )
    payload = response.get_json()
    print(payload)
    for field, value_type in [
        ('conf_speech_acts', list),
        ('speech_acts', list),
        ('utterances', list)
    ]:
        assert field in payload
        assert isinstance(payload[field], value_type)
