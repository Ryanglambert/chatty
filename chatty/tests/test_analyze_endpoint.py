import pytest

from flask import jsonify, request


def test_analyze(client):
    response = request.post(
        path='/analyze',
        content_type="application/json",
        data={'text': 'test __eou__ another'}
    )
    assert False