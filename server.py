from flask import Flask
from flask import request, make_response
import os
# from model import recognize

os.environ['FLASK_DEBUG'] = str(1)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def root():

    if request.method == 'POST':
        print(request.get_json())
        print(request.get_data())

        response_text = 'OK'
        resp = make_response(response_text)
        resp.headers['access-control-allow-origin'] = '*'
        return resp
    if request.method == 'GET':
        with open('index.html') as f:
            index_page = f.read()
        return index_page


if __name__ == '__main__':
    app.run(host='0.0.0.0')
