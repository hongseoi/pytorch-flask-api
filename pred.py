pred_class = output.argmax()
pred_class2 = output.argsort()[-2]

probs = output.softmax(dim=-1)
probs1 = probs[pred_class]
probs2 = probs[pred_class2]

print("pred_class ?", pred_class)
print(probs.shape)
print('probs:', probs1)
print('probs2:', probs2)
print("pred_class2 ?", pred_class2)

print("pred_class type?", type(pred_class))

다음과 같이 코드를 수정하면 됩니다.

Python
def predict(input_data):
    """
    입력 데이터에 대한 예측을 수행합니다.

    Args:
        input_data: 입력 데이터

    Returns:
        예측 결과
    """

    # 모델을 실행합니다.
    output = model(input_data)

    # 가장 확률 높은 클래스를 찾습니다.
    pred_class = output.argmax()

    # 두 번째로 확률 높은 클래스를 찾습니다.
    pred_class2 = output.argsort()[-2]

    # 확률을 찾습니다.
    probs = output.softmax(dim=-1)
    probs1 = probs[pred_class]
    probs2 = probs[pred_class2]

    # 예측을 출력합니다.
return jsonify({
        "predict_class": classes[pred_class],
        "predict_class2": classes[pred_class2],
        "prob1": probs1,
        "prob2": probs2,
    })