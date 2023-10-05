from flask import Flask, request, jsonify
import cv2
from utils import load_b64_image, dump_b64_image, mosaic_face

app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({
        "status": "ok",
        "message": "Flask Image Retouching Sample"
    }), 200

@app.route("/face-blur", methods=['POST'])
def face_blur():
    """
    指定された画像の中から、顔の部分をぼかしてBase64形式で返す
    リクエスト時にJSONを受け取り、image_url(必須)でBase64形式の画像を受け取る
    """

    # リクエストのJSONを取得
    reqJSON = request.json

    # image_urlが指定されていない場合はエラーを返す
    if not "image_url" in reqJSON:
        return jsonify({
            "status": "error",
            "message": "image_url is required"
        }), 400

    # image_urlから画像を取得
    img = None
    try:
        img = load_b64_image(reqJSON["image_url"])
    except:
        return jsonify({
            "status": "error",
            "message": "this image_url is invalid"
        }), 400

    # 顔認識を行う
    img = mosaic_face(img, ratio=0.5)

    # Base64形式に変換
    img_base64 = dump_b64_image(img)

    return jsonify({
        "status": "ok",
        "image": img_base64
    }), 200

if __name__ == '__main__':
    app.run()
