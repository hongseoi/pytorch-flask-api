import requests

# POST 요청
with open('./sample.jpg', 'rb') as file:
    resp = requests.post("http://localhost:5000/predict", files={"file": file})

# 응답 확인
if resp.status_code == 200:
    result = resp.json()
    print("Top 2 Classes:", result["top2_class"])
    print("Top 2 Percentages:", result["top2_percent"])
else:
    print("Error:", resp.status_code, resp.text)