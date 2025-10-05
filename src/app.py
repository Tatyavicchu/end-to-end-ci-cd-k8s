import joblib
import pandas as pd
from flask import Flask, request, jsonify
from logging_config import get_logger


logger = get_logger('app', level=20)
app = Flask(__name__)
model = None


def load_model(path='models/model.joblib'):
    global model
    model = joblib.load(path)
    logger.info('Model loaded')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    # expecting JSON with feature columns or array
    df = pd.DataFrame(data)
    preds = model.predict(df).tolist()
    logger.info(f'Prediction request: {len(df)} rows')
    return jsonify({'predictions': preds})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)