import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, GRU, Dense, TimeDistributed
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

def load_data():
    # 데이터를 로드하고 전처리
    X = np.random.randn(1000, 128, 64, 1)  # 1000개의 샘플, 임의의 데이터
    y = np.random.randint(0, 10, size=(1000, 128, 10))  # 10개의 클래스, 원-핫 인코딩된 레이블
    return X, y

from google.colab import drive
drive.mount('/content/drive')

def add_watermark_to_waveform(waveform, watermark_pattern):
    pattern_length = len(watermark_pattern)
    waveform_length = len(waveform)
    repeated_pattern = np.tile(watermark_pattern, waveform_length // pattern_length + 1)
    return waveform + repeated_pattern[:waveform_length]

def create_crnn_model(input_shape, num_classes):
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(32, (3, 3), padding='same', activation='relu')(input_data)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Conv2D(64, (3, 3), padding='same', activation='relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)

    inner_shape = inner.shape  # (batch_size, 32, 16, 64)
    _, h, w, c = inner_shape

    inner = Reshape(target_shape=(h, w * c))(inner)
    inner = GRU(128, return_sequences=True)(inner)
    inner = GRU(128, return_sequences=True)(inner)
    y_pred = TimeDistributed(Dense(num_classes, activation='softmax'))(inner)

    model = Model(inputs=input_data, outputs=y_pred)
    model.summary()
    return model

# 데이터 로드
X, y = load_data()

# 데이터셋을 학습용과 검증용으로 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 예제 워터마크 패턴
watermark_pattern = np.sin(np.linspace(0, 2 * np.pi, 100))

# 학습 데이터에 워터마크 추가
X_train_watermarked = np.array([add_watermark_to_waveform(x.flatten(), watermark_pattern).reshape(128, 64, 1) for x in X_train])

# 검증 데이터에 워터마크 추가 (필요에 따라 추가)
X_val_watermarked = np.array([add_watermark_to_waveform(x.flatten(), watermark_pattern).reshape(128, 64, 1) for x in X_val])

# 모델 입력 형태
input_shape = (128, 64, 1)
num_classes = 10

# 모델 생성 및 컴파일
model = create_crnn_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# y_train과 y_val의 모양 확인하고 필요한 경우 수정
y_train = np.array(y_train)  # Ensure y_train is a numpy array
y_val = np.array(y_val)      # Ensure y_val is a numpy array

# y_train과 y_val 모양 자르기 --> Conv2D랑 MaxPooling2D 레이어의 출력 모양 시퀀스 길이 맞춰야해서
y_train = y_train[:, :32, :]  # (batch_size, 32, num_classes)
y_val = y_val[:, :32, :]      # (batch_size, 32, num_classes)

print("Shape of y_train:", y_train.shape)  # Check the shape of y_train
print("Shape of y_val:", y_val.shape)      # Check the shape of y_val

# 모델 학습
history = model.fit(X_train_watermarked, y_train, validation_data=(X_val_watermarked, y_val), epochs=2)
