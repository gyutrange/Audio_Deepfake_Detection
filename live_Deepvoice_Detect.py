import pyaudio
import numpy as np
import librosa
import tensorflow as tf

# 오디오 스트림 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def capture_audio(duration=5):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = []
    print("Recording...")
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))
    
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio = np.hstack(frames)
    return audio

def extract_features_from_audio(audio, sr=16000):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

# 오디오 캡처
audio = capture_audio(duration=5)

# 특징 추출
features = extract_features_from_audio(audio)
features = features[..., np.newaxis]  # 모델 입력 형태로 변환

# 모델 로드 (학습된 모델을 저장하고 로드하는 예제)
model = tf.keras.models.load_model('crnn_model.h5')

# 예측
features = np.expand_dims(features, axis=0)  # 배치 차원 추가
prediction = model.predict(features)

# 결과 출력
if prediction > 0.5:
    print("The audio is detected as a deepfake.")
else:
    print("The audio is detected as an original.")

