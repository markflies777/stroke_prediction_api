from flask import Flask, request
from flask_cors import CORS, cross_origin
from serve import prediction
import pandas as pd
from http.server import HTTPServer, SimpleHTTPRequestHandler

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

stroke_predict = prediction()


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    input_data = request.json
    input_df = pd.DataFrame.from_dict([input_data])
    output_data = stroke_predict(input_df)
    return str(output_data)


if __name__ == '__main__':
    app.run(debug=True)
