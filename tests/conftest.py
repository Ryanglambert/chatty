import os
import pytest

from chatty.conf import conf
from chatty.api.api import create_app


@pytest.fixture(scope='session')
def app():
    # Create an `app` with testing configuration.
    # os.environ['JUSTICE_CONFIG'] = 'config/justice_testing.py'
    _app = create_app()

    with _app.app_context():
        yield _app


@pytest.fixture(scope='function')
def client(app):
    with app.test_client() as _client:
        yield _client
