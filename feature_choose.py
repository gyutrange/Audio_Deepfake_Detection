import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 파일 경로 설정
base_path = './'
bonafide_path = os.path.join(base_path, 'watermark_bonafide')
spoof_path = os.path.join(base_path, 'watermark_spoof')

# 오디오 파일 경로 리스트 생성
def get_file_paths(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.wav')]

original_file_paths = get_file_paths(bonafide_path)  # 원본 오디오 파일 경로 리스트
fake_file_paths = get_file_paths(spoof_path)        # 딥페이크 오디오 파일 경로 리스트

# 오디오 증강 함수
def augment_audio(audio, sr):
    augmented_data = []
    # 피치 변조
    pitch_shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=2)
    augmented_data.append(pitch_shifted)
    # 시간 확장
    time_stretched = librosa.effects.time_stretch(y=audio, rate=1.1)
    augmented_data.append(time_stretched)
    # 노이즈 추가
    noise = np.random.randn(len(audio))
    audio_with_noise = audio + 0.005 * noise
    augmented_data.append(audio_with_noise)
    return augmented_data

# 특징 추출 함수
def extract_features(file_path, sr=16000, augment=False, feature_type='mel'):
    audio, _ = librosa.load(file_path, sr=sr)
    if augment:
        augmented_audios = augment_audio(audio, sr)
        features = []
        for aug_audio in augmented_audios:
            if feature_type == 'mel':
                feature = librosa.feature.melspectrogram(y=aug_audio, sr=sr, n_mels=128)
                feature = librosa.power_to_db(feature)
            elif feature_type == 'mfcc':
                feature = librosa.feature.mfcc(y=aug_audio, sr=sr, n_mfcc=13)
            elif feature_type == 'spectral_contrast':
                feature = librosa.feature.spectral_contrast(y=aug_audio, sr=sr)
            elif feature_type == 'chromagram':
                feature = librosa.feature.chroma_stft(y=aug_audio, sr=sr)
            elif feature_type == 'zcr':
                feature = librosa.feature.zero_crossing_rate(y=aug_audio)
            features.append(feature)
        return features
    else:
        if feature_type == 'mel':
            feature = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            feature = librosa.power_to_db(feature)
        elif feature_type == 'mfcc':
            feature = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        elif feature_type == 'spectral_contrast':
            feature = librosa.feature.spectral_contrast(y=audio, sr=sr)
        elif feature_type == 'chromagram':
            feature = librosa.feature.chroma_stft(y=audio, sr=sr)
        elif feature_type == 'zcr':
            feature = librosa.feature.zero_crossing_rate(y=audio)
        return [feature]

# 데이터 로드 및 특징 추출 함수
def load_data(file_paths, sr=16000, augment=False, fixed_shape=(128, 128), feature_type='mel'):
    features = []
    for file_path in file_paths:
        print(file_path)
        extracted_features = extract_features(file_path, sr=sr, augment=augment, feature_type=feature_type)
        for feature in extracted_features:
            # 패딩 또는 자르기를 통해 특징 배열을 고정 크기로 만듭니다.
            mel_height, mel_width = feature.shape
            target_height, target_width = fixed_shape

            if mel_height < target_height:
                # 높이가 부족할 경우 패딩
                feature = np.pad(feature, ((0, target_height - mel_height), (0, 0)), mode='constant')
            elif mel_height > target_height:
                # 높이가 많을 경우 자르기
                feature = feature[:target_height, :]

            if mel_width < target_width:
                # 폭이 부족할 경우 패딩
                feature = np.pad(feature, ((0, 0), (0, target_width - mel_width)), mode='constant')
            elif mel_width > target_width:
                # 폭이 많을 경우 자르기
                feature = feature[:, :target_width]

            features.append(feature)
    return np.array(features)

# CRNN 모델 생성 함수
def create_crnn_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Flatten 레이어 추가
    model.add(tf.keras.layers.Flatten())
    # Reshape layer를 위해 정확한 output shape 계산
    model.add(tf.keras.layers.Reshape((input_shape[0] // 4, -1)))  # Flatten 후 Reshape. second dimension을 동적으로 계산
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 특징 종류를 지정해가며 학습 진행
feature_types = ['mel', 'mfcc', 'spectral_contrast', 'chromagram', 'zcr']
losses = []
accuracies = []

for feature_type in feature_types:
    print(f"Processing feature: {feature_type}")
    original_features = load_data(original_file_paths, augment=True, feature_type=feature_type)
    fake_features = load_data(fake_file_paths, augment=True, feature_type=feature_type)

    # 특징과 라벨 결합
    features = np.concatenate((original_features, fake_features), axis=0)
    labels = [0] * len(original_features) + [1] * len(fake_features)
    labels = np.array(labels)

    # 데이터셋 분할 (훈련, 검증)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 데이터 형태 변환 (모델 입력 형태에 맞게)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    # 모델 생성 및 학습
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (height, width, channels)
    model = create_crnn_model(input_shape)

    # EarlyStopping 콜백 추가
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 모델 학습
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=early_stopping)

    # 모델 평가
    loss, accuracy = model.evaluate(X_val, y_val)
    losses.append((feature_type, loss))
    accuracies.append((feature_type, accuracy))

    # 모델 저장
    model.save(f"crnn_model_{feature_type}.h5")

# 결과 출력
print("\nFinal Results:")
for i, feature_type in enumerate(feature_types):
    print(f"{feature_type} - Loss: {losses[i][1]}, Accuracy: {accuracies[i][1]}")
# Final Results:
# mel - Loss: 0.09578830748796463, Accuracy: 0.9658595323562622
# mfcc - Loss: 0.2109735906124115, Accuracy: 0.9123036861419678
# spectral_contrast - Loss: 0.5220527052879333, Accuracy: 0.7734511494636536
# chromagram - Loss: 0.3546830415725708, Accuracy: 0.8468586206436157
# zcr - Loss: 0.34841856360435486, Accuracy: 0.8498036861419678