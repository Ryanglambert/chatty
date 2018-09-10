import json
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

    slack_input_model = api.model('Input', {
        "text": fields.String("Text of your conversation from Slack", required=True)
    })

    speech_conf_fields = {}
    speech_conf_model = api.model('Speech Confs', {
        'commissive': fields.Float(),
        'directive': fields.Float(),
        'inform': fields.Float(),
        'question': fields.Float()
    })


    output_model = api.model('Output', {
        "utterances": fields.List(fields.String),
        # Fields don't provide anything useful yet
        # "sentiment": fields.List(fields.String),
        # "conf_sentiment": fields.List(fields.Float),
        "speech_acts": fields.List(fields.String),
        "conf_speech_acts": fields.List(fields.Nested(speech_conf_model)),
        # "next_sentiment": fields.String(),
        # "conf_next_sentiment": fields.Float()
    })

    @api.route('/analyze_slack')
    class ChatAnalyzer(Resource):
        @api.expect(slack_input_model)
        @api.marshal_with(output_model)
        def post(self):
            """
            Analyze some Text from slack. Just copy and paste directly from slack (I use the datetime headings to parse between utterances)
            """
            text = json.loads(request.data, strict=False)['text']
            return analyze(text), 201

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=conf['api']['debug'], port=conf['api']['port'])
