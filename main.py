from flask import Flask, request
from serve import prediction
import pandas as pd

app = Flask(__name__)

stroke_predict = prediction()


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame.from_dict([input_data])
    output_data = stroke_predict(input_df)
    return str(output_data)


if __name__ == '__main__':
    app.run(debug=True)
