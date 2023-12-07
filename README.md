# PyTorch Flask API

This repo contains a sample code to show how to create a Flask API server by deploying our PyTorch model. This is a sample code which goes with [tutorial](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html).

If you'd like to learn how to deploy to Heroku, then check [this repo](https://github.com/avinassh/pytorch-flask-api-heroku).


## API 정의
- 이미지가 포함된 file 매개변수를 HTTP POST로 /predict에 request하고 다음과 같은 예측 결과를 JSON 형태로 response함

```
{"top1_class":'a', "top2_percent":0.3, "top2_class":'b', "top2_percent":0.1}
``` 


## How to 

Install the dependencies:

```
    conda create --name ENVNAME python==3.9
    conda activate ENVNAME
    git clone REPOURL
    cd REPOFOLDER
    pip install -r requirements.txt

```


Run the Flask server:
```
    flask run

```


From another tab, send the image file in a request:

    curl -X POST -F file=@cat_pic.jpeg http://localhost:5000/predict


## License

The mighty MIT license. Please check `LICENSE` for more details.
