from flask import Flask
from flask import request, make_response
import os
import json
# from model import recognize

os.environ['FLASK_DEBUG'] = str(1)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def root():

    resp = None

    if request.method == 'POST':
        array = json.loads(request.get_data().decode('utf-8'))['array']
        print(array[0])

        response_text = 'OK'
        resp = make_response(response_text)

    if request.method == 'GET':
        with open('index.html') as f:
            index_page = f.read().encode('utf-8')
        resp = make_response(index_page)

    resp.headers['access-control-allow-origin'] = '*'
    return resp


if __name__ == '__main__':
    app.run()
