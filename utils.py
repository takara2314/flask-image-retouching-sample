import numpy as np
import cv2
from base64 import b64decode, b64encode

# 顔認識用のxmlファイル
face_cascade_path = "haarcascade_frontalface_default.xml"

# Base64形式の画像データ -> OpenCV形式の画像データ
def load_b64_image(b64):
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    return cv2.imdecode(
        np.frombuffer(
            b64decode(b64.encode()),
            np.uint8
        ),
        cv2.IMREAD_ANYCOLOR
    )

# OpenCV形式の画像データ -> Base64形式の画像データ
def dump_b64_image(img):
    return b64encode(cv2.imencode(".png", img)[1]).decode("utf-8")

# モザイク処理
def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

# 指定した範囲にモザイクをかける
def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

# 顔認識して顔部分にモザイクをかける
def mosaic_face(img, ratio=0.1):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(src_gray)

    for x, y, w, h in faces:
        img = mosaic_area(img, x, y, w, h, ratio)

    return img
