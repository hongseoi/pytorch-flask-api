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

# 모델 로드
state_dict1 = torch.load("./models/s1_binary_class.pth", map_location=torch.device('cpu'))
state_dict2 = torch.load("./models/s3_disease_class.pth", map_location=torch.device('cpu'))

# 모델 불러오고 evaluation 상태로 지정
model1 = md.Plant_Disease_Model()
model2 = md.Plant_Disease_Model2()

model1.load_state_dict(state_dict1)
model1.eval()

model2.load_state_dict(state_dict2)
model2.eval()

transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.ToTensor(),
])

# 예측 결과 불러오기
def get_prediction(image_bytes):
    """
    - input: image_bytes
    - return :
        - top_2_classes: ["a", "b"]
        - top_2_probs: [32, 12]
    """
    tensor = pdm.transform_image(image_bytes=image_bytes)
    top1_lst = []
    top2_lst = []
    probs_lst = []

    with torch.no_grad():
       image = Image.open("test2.jpg")
       image = transform(image)
       #image = image[:,:128,:]
       image = torch.unsqueeze(image, 0)
       output1 = model1(image)
       pred_class = output1.argmax()
       print(pred_class)
       
       if pred_class.item() == 0: # 식물 아닌 경우print("식물 아님")
        print("식물아님")

        top1_class, top2_class = "식물아님"
        top_probs = [100, 100]
        print(top1_class, top2_class)

       if pred_class.item() == 1: # 식물인경우
        print("식물임")

        output2 = model2(image)
        _, preds = torch.topk(output2, 2)
        probs = torch.nn.functional.softmax(output2, dim=1)[0,  preds[0]]
        print(preds, probs)
        print(probs.shape)

        top_classes = preds.tolist()
        top_probs = probs.tolist()

        print(top_classes, top_probs)

        top1_class = classes[top_classes[0][0]]
        top2_class = classes[top_classes[0][1]]

        print(top1_class, top2_class)

    top1_lst.append(top1_class)
    top2_lst.append(top2_class)
    probs_lst.append(top_probs)

    return top1_lst, top2_lst, probs_lst


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class1_name, class2_name, class_probs = get_prediction(image_bytes=img_bytes)
        return jsonify({"top1_class":class1_name, "top2_class":class2_name, "top1_percent":class_probs[0], "top2_percent":class_probs[1]})

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


