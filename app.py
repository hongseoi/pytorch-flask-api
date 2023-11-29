import io
import json
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from model import Plant_Disease_Model2 as pdm
import model as md
from flask import Flask, render_template
import torch

app = Flask(__name__)

# 모델 클래스
classes = ['bean__bean_spot', 'bean__blight', 'bean__brown_spot', 'bean__healthy', 'corn__common_rust', 'corn__gray_spot', 'corn__healthy', 'green_onion__black_spot', 'green_onion__downy_mildew', 'green_onion__healthy', 'green_onion__rust', 'lectuce__downy_mildew', 'lectuce__drop', 'lectuce__healthy', 'pepper__anthracnose', 'pepper__healthy', 'pepper__powdery_mildew', 'potato__Early_Blight', 'potato__healthy', 'potato__late_Blight', 'potato__soft_rot', 'pumpkin__healthy', 'pumpkin__leaf_mold', 'pumpkin__mosaic', 'pumpkin__powdery_mildew', 'radish__black_spot', 'radish__downy_mildew', 'radish__healthy']
# 모델을 훈련할 때 사용한 클래스 수가 28이라면,
print(len(classes))

# 모델 불러오고 evaluation 상태로 지정
model = md.Plant_Disease_Model2()
model.eval()


# 예측 결과 불러오기
def get_prediction(image_bytes):
    """
    - input: image_bytes
    - return :
        - top_2_classes: ["a", "b"]
        - top_2_probs: [32, 12]
    """
    tensor = pdm.transform_image(image_bytes=image_bytes)
    outputs = model(tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# 상위 2개 클래스 및 확률 출력
    top2_probabilities, top2_classes = torch.topk(probabilities, 2)
    top2_classes = top2_classes.tolist()    # 텐서를 리스트로

    print("class:", top2_classes, "probs:", top2_probabilities)
    for i in range(top2_probabilities.size(0)):
        class_idx = top2_classes[i]
        class_name = classes[class_idx]
        probability = top2_probabilities[i].item()
    print(f"상위 {i + 1} 클래스: {class_name}, 확률: {probability:.2%}")
    return class_name, probability
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name, class_probs = get_prediction(image_bytes=img_bytes)
        return jsonify({"top2_class":class_name, "top2_percent":class_probs})

# 예시(지워도됨)
with open("./test1.jpg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))

# # POST 요청
# import requests
# resp = requests.post("http://localhost:5000/predict",
#                      files={"file": open('sample.jpg','rb')})

# flask 서버 실행 코드
if __name__ == '__main__':
    app.run()


