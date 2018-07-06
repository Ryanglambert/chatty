from flask import Flask, jsonify
from flask_restplus import Api, Resource, cors

from conf import conf

app = Flask(__name__)
api = Api(app)

@api.route('/chatty/')
class HelloWorld(Resource):
    @cors.crossdomain(origin="*")
    def get(self):
        return jsonify({'hello': ['world', 'world2']})


if __name__ == '__main__':
    app.run(debug=conf['chatty_rest']['debug'])
