import os
import traceback
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from flask import Flask, jsonify, request
from executors.cvdnn_inferrer import Campus_inferrer
app = Flask(__name__)

APP_ROOT = os.getenv('APP_ROOT', '/infer')
HOST = "127.0.0.1"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 5000))

c_net = Campus_inferrer()


@app.route(APP_ROOT, methods=["POST"])
def infer():
    data = request.json
    image = data['image']
    return c_net.infer(image)


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if __name__ == 'main':
    app.run(host=HOST, port=PORT_NUMBER)


"""import os
import traceback
import sys
sys.path.append(os.path.join(os.path.dirname(file),'../'))
from flask import Flask, jsonify, request
from executors.unet_inferrer import UnetInferrer
app = Flask(name)

APP_ROOT = os.getenv('APP_ROOT', '/infer')
HOST = "127.0.0.1"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 5000))

u_net = UnetInferrer()


@app.route(APP_ROOT, methods=["POST"])
def infer():
    data = request.json
    image = data['image']
    return u_net.infer(image)


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())


if name == 'main':
    app.run(host=HOST, port=PORT_NUMBER)


"""