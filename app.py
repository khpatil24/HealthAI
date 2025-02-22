from flask import Flask, request, jsonify
import numpy as np
import pickle
#from main import model, dataset, process_user_input  # Ensure 'model', 'dataset', and 'process_user_input' exist in main.py

app = Flask(__name__)

model_path = 'model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# API route to process input
@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    user_text = data.get('text', '')
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400
    response_vector = model.process_user_input(user_text)
    return jsonify({'symptom_vector': response_vector})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
