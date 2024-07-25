import os
import numpy as np
import librosa
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

def extract_features_from_audio(audio, sr=16000):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    if log_mel_spectrogram.shape[1] < 128:
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, 128 - log_mel_spectrogram.shape[1])), mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :128]
    return log_mel_spectrogram

def predict_audio_file(model, file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    features = extract_features_from_audio(audio, sr=sr)
    features = features[..., np.newaxis]
    features = np.expand_dims(features, axis=0)
    
    prediction = model.predict(features)
    
    deepfake_probability = prediction[0][0]
    if deepfake_probability > 0.5:
        return "deepfake", deepfake_probability
    else:
        return "original", 1 - deepfake_probability

def get_wav_files(directory):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

def evaluate_model_on_directory(model, directory):
    wav_files = get_wav_files(directory)
    results = {"deepfake": 0, "original": 0}
    
    for file_path in wav_files:
        print(f"\nPredicting file: {file_path}")
        label, probability = predict_audio_file(model, file_path)
        
        if label == "deepfake":
            results["deepfake"] += 1
            print(f"The audio in {file_path} is detected as a deepfake with {probability:.3%} probability.")
        else:
            results["original"] += 1
            print(f"The audio in {file_path} is detected as an original with {probability:.3%} probability.")
    
    return results, len(wav_files)

# 모델 로드 (학습된 모델을 저장하고 로드하는 예제)
model = tf.keras.models.load_model('/content/drive/MyDrive/crnn_model.h5')

# 예측할 디렉토리 설정
directory = '/content/drive/MyDrive/61_watermark_bonafide/'

# 디렉토리 내의 WAV 파일에 대해 예측 실행 및 결과 요약
results, total_files = evaluate_model_on_directory(model, directory)
print(f"\nTotal WAV files: {total_files}")
print(f"Deepfake Count: {results['deepfake']}, Original Count: {results['original']}")
