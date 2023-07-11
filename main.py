
from flask import Flask, request
from detect import detect
app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def home():
    file = request.files.get('image')
    path = f'uploads/{file.filename}'
    # file.save(path)
    result = detect(path)

    if result['class_name'] == 'UnKnown':
        res = f"{result['class_name']}"
    else:
        res = f"{result['class_name']} - Score: {result['predicted_prob']}%"
    return res


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
