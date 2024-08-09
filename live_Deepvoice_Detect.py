import yt_dlp
import librosa
import tensorflow as tf
import numpy as np
import os
import scipy.io.wavfile as wav
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
import time
import random
from tqdm import tqdm

# TensorFlow 모델 로드
model = tf.keras.models.load_model('noAUG_model_v2.h5')

# RSA 키 파일 경로
PRIVATE_KEY_FILE = "private.pem"


def extract_features_from_audio(audio, sr=16000):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    if log_mel_spectrogram.shape[1] < 128:
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, 128 - log_mel_spectrogram.shape[1])),
                                     mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :128]
    return log_mel_spectrogram


def predict_audio_file(model, audio, sr=16000):
    features = extract_features_from_audio(audio, sr=sr)
    features = features[..., np.newaxis]
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(features)
    deepfake_probability = prediction[0][0]
    if deepfake_probability > 0.5:
        return "deepfake", deepfake_probability
    else:
        return "original", 1 - deepfake_probability


def load_watermark_bits(filename):
    with open(filename, 'r') as file:
        watermark_bits = file.read().strip()
    return watermark_bits


def embed_watermark_spread_spectrum(audio, watermark_bits, rate):
    audio = audio.astype(np.float32)

    watermark_bits = np.array(list(map(int, watermark_bits)))

    # 강도 조절을 위한 시드 조정
    np.random.seed(0)
    spread_sequence = np.random.choice([1, -1], size=(len(watermark_bits), len(audio)))

    # 워터마크 강도 조절 (여기서 조절 가능)
    watermark_strength = 0.0001

    for i, bit in enumerate(watermark_bits):
        if bit == 1:
            audio += watermark_strength * spread_sequence[i]

    return np.int16(audio / np.max(np.abs(audio)) * 32767)


def download_audio(youtube_url, output_path='audio.wav'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,  # Enable output for debugging
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading audio...")
            ydl.download([youtube_url])
        print(f"Downloaded audio to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def split_and_analyze_audio(file_path, segment_duration=10):
    # Load the entire audio file
    audio, sr = librosa.load(file_path, sr=16000)

    # Calculate the number of segments
    total_duration = len(audio) / sr
    num_segments = int(np.ceil(total_duration / segment_duration))

    print("Analyzing audio...")
    # Load watermark bits from file
    watermark_bits = load_watermark_bits('watermark_bits.txt')
    print('Load bits success')

    # Analyze each segment
    for i in tqdm(range(num_segments), desc='Analyzing Segments'):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        segment = audio[start_sample:end_sample]
        # Embed watermark into segment
        watermarked_segment = embed_watermark_spread_spectrum(segment, watermark_bits, sr)

        # Save the watermarked segment temporarily
        temp_file = f'temp_segment_{i}.wav'
        wav.write(temp_file, sr, watermarked_segment)

        # Load the watermarked segment for prediction
        watermarked_audio, _ = librosa.load(temp_file, sr=16000)
        label, probability = predict_audio_file(model, watermarked_audio, sr=sr)
        print(f"Segment {i + 1}/{num_segments}: Prediction: {label} with probability: {probability:.3%}")

        # Clean up temporary file
        os.remove(temp_file)


def process_youtube_audio(youtube_url):
    download_audio(youtube_url, 'audio.wav')

    split_and_analyze_audio('audio.wav', segment_duration=10)

youtube_url = 'https://www.youtube.com/watch?v=k8X_Em-NQn0'  # Replace with your YouTube URL

# 오디오 다운로드, 워터마크 삽입 및 분석
process_youtube_audio(youtube_url)
~
