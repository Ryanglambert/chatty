from flask import Flask, jsonify
from flask_restplus import Api, Resource, cors


app = Flask(__name__)
api = Api(app)

@api.route('/hello')
class HelloWorld(Resource):
    @cors.crossdomain(origin="*")
    def get(self):
        return jsonify({'hello': ['world', 'world2']})


if __name__ == '__main__':
    app.run(debug=True)
