import os
import pytest

from chatty.conf import conf


@pytest.fixture
def app():
    from chatty.api.api import create_app
    app = create_app()
    app.run(debug=False)
    yield app
