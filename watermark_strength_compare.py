import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from google.colab import drive

#학습데이터는 실험용으로 100개 사용
#watermark strength = 0.001 0.0001사용(0.14 0.97)

# Google Drive 연결
drive.mount('/content/drive')

# 데이터 파일 경로 설정
base_path = '/content/drive/MyDrive/data/'
watermarked_0001_path = os.path.join(base_path, 'watermark00001_bonafide')
watermarked_00001_path = os.path.join(base_path, 'watermark0001_bonafide')

# 오디오 파일 경로 리스트 생성
def get_file_paths(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.wav')]

watermarked_00001_file_paths = get_file_paths(watermarked_00001_path)
watermarked_0001_file_paths = get_file_paths(watermarked_0001_path)

# 특징 추출 함수 (증강 없음)
def extract_features(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

# 데이터 로드 및 특징 추출 함수
def load_data(file_paths, sr=16000, fixed_shape=(128, 128)):
    features = []
    for file_path in file_paths:
        feature = extract_features(file_path, sr=sr)
        mel_height, mel_width = feature.shape
        target_height, target_width = fixed_shape

        if mel_height < target_height:
            feature = np.pad(feature, ((0, target_height - mel_height), (0, 0)), mode='constant')
        elif mel_height > target_height:
            feature = feature[:target_height, :]

        if mel_width < target_width:
            feature = np.pad(feature, ((0, 0), (0, target_width - mel_width)), mode='constant')
        elif mel_width > target_width:
            feature = feature[:, :target_width]

        features.append(feature)
    return np.array(features)

# 워터마크 오디오의 특징 추출
watermarked_00001_features = load_data(watermarked_00001_file_paths)
watermarked_0001_features = load_data(watermarked_0001_file_paths)

# 라벨 설정 (워터마크 0.00001: 0, 워터마크 0.0001: 1)
watermarked_00001_labels = [0] * len(watermarked_00001_features)
watermarked_0001_labels = [1] * len(watermarked_0001_features)

# 특징과 라벨 결합
features = np.concatenate((watermarked_00001_features, watermarked_0001_features), axis=0)
labels = np.array(watermarked_00001_labels + watermarked_0001_labels)

# 데이터셋 분할
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# 데이터 형태 변환 (모델 입력 형태에 맞게)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# CRNN 모델 생성 함수
def create_crnn_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Reshape((input_shape[0] // 4, -1)))  # Flatten 후 Reshape
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))  # 2개의 클래스 (워터마크 0.05, 워터마크 0.1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 모델 생성 및 학습
def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (height, width, channels)
    model = create_crnn_model(input_shape)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

    # 모델 저장
    model.save('/content/crnn_model.h5')

    # 모델 평가
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    return loss, accuracy

# 모델 학습 및 평가
loss, accuracy = train_and_evaluate_model(X_train, y_train, X_val, y_val)

# 데이터셋 별 성능 비교
print(f"Combined Data - Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# 워터마크 0.05 데이터만 사용
X_train_00001, X_val_00001, y_train_00001, y_val_00001 = train_test_split(watermarked_00001_features, watermarked_00001_labels, test_size=0.2, random_state=42)
X_train_00001 = X_train_00001[..., np.newaxis]
X_val_00001 = X_val_00001[..., np.newaxis]

loss_00001, accuracy_00001 = train_and_evaluate_model(X_train_00001, y_train_00001, X_val_00001, y_val_00001)
print(f"Watermarked 0.05 Data - Validation Loss: {loss_00001}, Validation Accuracy: {accuracy_00001}")

# 워터마크 0.1 데이터만 사용
X_train_0001, X_val_0001, y_train0001, y_val_0001 = train_test_split(watermarked_0001_features, watermarked_0001_labels, test_size=0.2, random_state=42)
X_train_0001 = X_train_0001[..., np.newaxis]
X_val_0001 = X_val_0001[..., np.newaxis]

loss_01, accuracy_01 = train_and_evaluate_model(X_train_0001, y_train_0001, X_val_0001, y_val_0001)
print(f"Watermarked 0.1 Data - Validation Loss: {loss_0001}, Validation Accuracy: {accuracy_0001}")