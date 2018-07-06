from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, cors, fields

from conf import conf

app = Flask(__name__)
api = Api(app)

input_model = api.model('Input', {
    "text": fields.String("Text of your conversation")
})

@api.route('/chatty/')
class HelloWorld(Resource):
    @cors.crossdomain(origin="*", methods="*", headers="*")
    def post(self):
        text = request.json['text']
        lines = text.split('\n')
        return jsonify({'lines': lines})

    def options(self):
        return {'Allow' : 'POST,GET,OPTIONS' }, 200, \
               {'Access-Control-Allow-Origin': '*', \
                'Access-Control-Allow-Methods' : 'POST,GET,OPTIONS',
                'Access-Control-Allow-Headers': '*'}


if __name__ == '__main__':
    app.run(debug=conf['chatty_rest']['debug'])
