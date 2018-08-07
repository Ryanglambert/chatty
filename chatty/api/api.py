import os
import numpy as np

from flask import Flask, jsonify, request
from flask_restplus import Namespace, Resource, fields, Api

from chatty.conf import conf


app = Flask(__name__, instance_relative_config=True)
# app.config.from_mapping(**conf['api'])
api = Api(app)

input_model = api.model('Input', {
    "text": fields.String("Text of your conversation", required=True)
})

output_model = api.model('Output', {
    "utterances": fields.List(fields.String),
    "sentiment": fields.List(fields.String),
    "next_sentiment": fields.String(),
    "confidence": fields.Float()
})

@api.route('/analyze')
class ChatAnalyzer(Resource):
    @api.expect(input_model)
    @api.marshal_with(output_model)
    def post(self):
        """Analyze some Text"""
        sentiment = ['happy', 'sad', 'surprised', 'disgust', 'fear', 'anger']
        text = request.get_json(
            force=True
        )['text']
        utterances = text.split('__eou__')
        return {
            'utterances': utterances,
            'sentiment': [np.random.choice(sentiment) for i in utterances],
            'next_sentiment': np.random.choice(sentiment),
            'confidence': np.random.rand()
        }, 201

if __name__ == '__main__':
    app.run(debug=True)
