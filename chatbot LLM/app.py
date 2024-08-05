from flask import Flask,jsonify,request,render_template
from llm import document_search
import os
import constants
from flask_cors import CORS, cross_origin
api_key = os.environ["OPENAI_API_KEY"] = constants.API_KEY

app = Flask(__name__)
cors = CORS(app)

data_path = 'data/'

@app.route('/')
def index():
    return render_template('query.html')

@app.route('/search', methods=['POST'])
def search_endpoint():
    try:
        query = request.json.get('query')
        result = document_search(api_key,data_path, query)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)