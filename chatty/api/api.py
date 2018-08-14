import os
import numpy as np

from flask import Flask, jsonify, request
from flask_restplus import Namespace, Resource, fields, Api

from chatty.analyze import analyze
from chatty.conf import conf


def create_app(config=conf['api']):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(**config)
    api = Api(app)

    input_model = api.model('Input', {
        "text": fields.String("Text of your conversation", required=True)
    })

    output_model = api.model('Output', {
        "utterances": fields.List(fields.String),
        "sentiment": fields.List(fields.String),
        "conf_sentiment": fields.List(fields.Float),
        "speech_acts": fields.List(fields.String),
        "conf_speech_acts": fields.List(fields.Float),
        "next_sentiment": fields.String(),
        "conf_next_sentiment": fields.Float()
    })

    @api.route('/analyze')
    class ChatAnalyzer(Resource):
        @api.expect(input_model)
        @api.marshal_with(output_model)
        def post(self):
            """Analyze some Text"""
            text = request.get_json(
                force=True
            )['text']
            return analyze(text), 201
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=conf['api']['debug'])
