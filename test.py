# 動作テスト

import requests
import cv2
from utils import load_b64_image, dump_b64_image

def test_face_blur(base_url):
    img = cv2.imread("test_input.png")
    img_base64 = dump_b64_image(img)

    res = requests.post(
        f"{base_url}/face-blur",
        headers={
            "Content-Type": "application/json"
        },
        json={
            "image_url": img_base64
        }
    )

    if res.status_code != 200:
        print(res.json())
        return

    resJSON = res.json()
    img_base64 = load_b64_image(resJSON["image"])

    cv2.imwrite("test_output.png", img_base64)


if __name__ == '__main__':
    test_face_blur("http://localhost:5000")
