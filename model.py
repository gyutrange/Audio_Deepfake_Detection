import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 오디오 파일 경로 리스트 준비
original_file_paths = [...]  # 원본 오디오 파일 경로 리스트
fake_file_paths = [...]      # 딥페이크 오디오 파일 경로 리스트
labels = [0] * len(original_file_paths) + [1] * len(fake_file_paths)

# 특징 추출 함수
def extract_features(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

# 데이터 로드 및 특징 추출
def load_data(file_paths, sr=16000):
    features = []
    for file_path in file_paths:
        features.append(extract_features(file_path, sr=sr))
    return np.array(features)

# 원본 오디오와 딥페이크 오디오의 특징 추출
original_features = load_data(original_file_paths)
fake_features = load_data(fake_file_paths)

# 특징과 라벨 결합
features = np.concatenate((original_features, fake_features), axis=0)
labels = np.array(labels)

# 데이터셋 분할 (훈련, 검증)
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# 데이터 형태 변환 (모델 입력 형태에 맞게)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

def create_crnn_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Reshape((input_shape[0] // 4, (input_shape[1] // 4) * 64)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = create_crnn_model(input_shape)

# 모델 학습
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=128, batch_size=30)

# 모델 평가
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# 예측
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)  # 이진 분류 결과로 변환

# 예측 결과 출력
if y_pred == 1:
    print("deepfake")
else:
    print("origin")

# 실제 라벨과 비교하여 정확도 계산
accuracy = accuracy_score(y_val, y_pred)
print(f"Prediction Accuracy: {accuracy}")

# 모델 저장
model.save('crnn_model.h5')
