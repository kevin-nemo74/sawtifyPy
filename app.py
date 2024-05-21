import os
import numpy as np
from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import torchaudio
from jiwer import cer

app = Flask(__name__)

# Load Wav2Vec 2.0 model and tokenizer
model_name = "facebook/wav2vec2-large-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def extract_transcription(file_path):
    # Load audio
    speech, rate = torchaudio.load(file_path)
    
    # Resample to 16000 Hz
    if rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
        speech = resampler(speech)
    
    # Tokenize and get logits
    input_values = tokenizer(speech.squeeze().numpy(), return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits
    
    # Decode the logits to get the transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

@app.route('/compare', methods=['POST'])
def compare():
    user_audio = request.files['user_audio']
    target_audio = request.files['target_audio']

    # Ensure the temporary directory exists
    temp_dir = '/tmp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    user_audio_path = os.path.join(temp_dir, user_audio.filename)
    target_audio_path = os.path.join(temp_dir, target_audio.filename)

    user_audio.save(user_audio_path)
    target_audio.save(target_audio_path)

    user_transcription = extract_transcription(user_audio_path)
    target_transcription = extract_transcription(target_audio_path)

    distance = cer(target_transcription, user_transcription)

    os.remove(user_audio_path)
    os.remove(target_audio_path)

    return jsonify({'distance': distance, 'user_transcription': user_transcription, 'target_transcription': target_transcription})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
