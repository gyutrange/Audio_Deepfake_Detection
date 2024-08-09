import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.io.wavfile as wav

def plot_frequency_spectrum(audio, rate, title):
    # FFT를 사용한 주파수 분석
    N = len(audio)
    yf = fft(audio)
    xf = np.linspace(0.0, rate / 2.0, N // 2)
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.grid()
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

def preprocess_audio(file_name):
    rate, audio = wav.read(file_name)
    if audio.dtype == np.int16:
        # 데이터 형식을 float32로 변환
        audio = audio.astype(np.float32)
        audio /= np.max(np.abs(audio))  # 정규화
    elif audio.dtype != np.float32:
        raise ValueError("Unsupported audio format")
    return rate, audio

# 원본 오디오 파일 읽기
rate_original, audio_original = wav.read('/content/81.wav')

# 워터마크가 삽입된 오디오 파일 읽기
rate_watermarked, audio_watermarked = wav.read('/content/watermarked_81.wav')

# 두 파일의 샘플링 주파수가 동일한지 확인
if rate_original != rate_watermarked:
    raise ValueError("Sampling rates of the audio files do not match!")

# 원본 오디오와 워터마크 삽입 오디오의 주파수 분석 플로팅
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plot_frequency_spectrum(audio_original, rate_original, 'Original Audio Frequency Spectrum')

plt.subplot(1, 2, 2)
plot_frequency_spectrum(audio_watermarked, rate_watermarked, 'Watermarked Audio Frequency Spectrum')

plt.tight_layout()
plt.show()
