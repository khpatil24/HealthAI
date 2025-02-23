from flask import Flask, request, jsonify
import pickle
from main import model, process_user_input 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   

model_path = 'model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# API route to process input (plain text)
@app.route('/process', methods=['POST'])
def process():
    user_text = request.data.decode('utf-8').strip()  # Read raw text input
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400
    response_vector = process_user_input(user_text)
    return jsonify({'symptom_vector': response_vector})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

