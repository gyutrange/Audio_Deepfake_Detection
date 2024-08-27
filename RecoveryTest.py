import numpy as np
import scipy.io.wavfile as wav
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
import time
import matplotlib.pyplot as plt
import reedsolo


# RSA 키 쌍 생성 및 저장
def generate_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()

    with open("private.pem", "wb") as f:
        f.write(private_key)

    with open("public.pem", "wb") as f:
        f.write(public_key)

    return "private.pem", "public.pem"


# 디지털 서명 생성
def create_signature(data, private_key_file):
    timestamp = str(int(time.time()))
    data_to_sign = data + "|" + timestamp

    hash_obj = SHA256.new(data_to_sign.encode('utf-8'))
    with open(private_key_file, "rb") as f:
        private_key = RSA.import_key(f.read())
    signature = pkcs1_15.new(private_key).sign(hash_obj)

    return data_to_sign, signature


# 문자열과 바이트를 비트로 변환
def to_bits(data):
    if isinstance(data, str):
        return ''.join(format(ord(char), '08b') for char in data)
    elif isinstance(data, bytes):
        return ''.join(format(byte, '08b') for byte in data)
    else:
        raise TypeError("Data must be of type str or bytes.")


# 비트를 문자열 또는 바이트로 변환
def bits_to_bytes(bits):
    bytes_list = [bits[i:i + 8] for i in range(0, len(bits), 8)]
    return bytes(int(b, 2) for b in bytes_list)


def bits_to_string(bits):
    bytes_data = bits_to_bytes(bits)
    return bytes_data.decode('utf-8', errors='ignore')


# 비프 소리 생성 (고주파 대역 사용)
def generate_beep(frequency=19000, duration=2, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    beep = 0.0001 * np.sin(2 * np.pi * frequency * t)  # 0.1로 진폭을 줄임 (낮은 음량)
    return np.int16(beep * 32767)


# LSB 기법으로 오디오에 디지털 서명 삽입
def embed_signature_lsb(audio, data_bits, spread_factor=2):
    audio = audio.astype(np.int16)
    num_bits = len(data_bits)

    for i in range(num_bits):
        sample_index = i * spread_factor
        if sample_index < len(audio):
            audio[sample_index] = (audio[sample_index] & ~1) | int(data_bits[i])

    return audio


# 오디오 파일에서 서명 추출
def extract_signature_lsb(audio, num_bits, spread_factor=2):
    extracted_bits = ""
    audio = audio.astype(np.int16)  # Ensure audio is in int16 format

    for i in range(num_bits):
        sample_index = i * spread_factor
        if sample_index < len(audio):
            extracted_bits += str(audio[sample_index] & 1)
        else:
            break  # Stop if we exceed the length of the audio array

    return extracted_bits


# 디지털 서명 검증
def verify_signature(data, signature, public_key_file):
    try:
        hash_obj = SHA256.new(data.encode('utf-8'))
        with open(public_key_file, "rb") as f:
            public_key = RSA.import_key(f.read())
        pkcs1_15.new(public_key).verify(hash_obj, signature)
        return True
    except (ValueError, TypeError):
        return False


def add_noise(audio, noise_level):
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise


def recover_signature_with_reed_solomon(damaged_bits, original_bits_length, n_data=255, k_data=223):
    # Reed-Solomon 에러 수정 코드의 n과 k 값 설정
    rs = reedsolo.RSCodec(n_data - k_data)
    original_data_bits = damaged_bits[:original_bits_length]

    try:
        # 원본 비트 수로 에러 수정
        recovered_bytes = rs.decode(bits_to_bytes(original_data_bits))
        recovered_bits = to_bits(recovered_bytes.decode('utf-8', errors='ignore'))
        return recovered_bits
    except reedsolo.ReedSolomonError:
        return None


def main():
    # 키 생성
    private_key_file, public_key_file = generate_keys()

    # 원본 데이터
    original_data = "This is a sample message for digital signature."

    # 디지털 서명 생성
    data_to_sign, original_signature = create_signature(original_data, private_key_file)

    # 데이터와 서명을 비트로 변환 및 결합
    data_bits = to_bits(data_to_sign)
    signature_bits = to_bits(original_signature)
    combined_bits = data_bits + signature_bits

    # 비프 소리 생성
    beep_duration = 0.2  # 비프 소리의 길이 (초)
    beep = generate_beep(duration=beep_duration)

    # 비프 소리 부분에 서명 삽입
    spread_factor = 2  # 압축된 삽입
    beep_with_signature = embed_signature_lsb(beep, combined_bits, spread_factor=spread_factor)

    # 원본 오디오 읽기
    input_audio_file = "bona_fide_directory/46.wav"
    rate, audio = wav.read(input_audio_file)

    # 스테레오 처리
    if audio.ndim == 2:
        audio = audio[:, 0]

    # 노이즈 단계 설정
    noise_levels = np.linspace(0.01, 0.1, 10)  # 10단계 노이즈 레벨
    recovery_rates = []
    recovered_signatures = []

    for noise_level in noise_levels:
        # 비프 소리에 노이즈 추가
        noisy_beep_with_signature = add_noise(beep_with_signature, noise_level)

        # 원본 오디오에 노이즈 추가
        noisy_audio = add_noise(audio, noise_level)

        # 비프 소리와 원본 오디오 결합
        final_audio = np.concatenate([noisy_beep_with_signature, noisy_audio])

        # 결과 오디오 저장
        output_audio_file = f"watermarked_audio_with_beep_noise_{noise_level:.2f}.wav"
        wav.write(output_audio_file, rate, final_audio)

        # 오디오 파일에서 서명 추출
        extracted_audio_rate, extracted_audio = wav.read(output_audio_file)
        beep_samples = len(noisy_beep_with_signature)  # 노이즈가 추가된 비프 소리 샘플 수 계산
        extracted_bits = extract_signature_lsb(extracted_audio[:beep_samples], len(combined_bits),
                                               spread_factor=spread_factor)

        # Reed-Solomon으로 서명 복구 시도
        recovered_bits = recover_signature_with_reed_solomon(extracted_bits, len(combined_bits))
        if recovered_bits:
            extracted_data_bits = recovered_bits[:len(data_bits)]
            extracted_signature_bits = recovered_bits[len(data_bits):]

            extracted_data = bits_to_string(extracted_data_bits)
            extracted_signature = bits_to_bytes(extracted_signature_bits)

            # 디지털 서명 검증
            is_valid = verify_signature(extracted_data, extracted_signature, public_key_file)
            recovery_rates.append(int(is_valid))
        else:
            recovery_rates.append(0)

    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, recovery_rates, marker='o', label='Recovery Rate with Noise and RS Code')
    plt.xlabel('Noise Level')
    plt.ylabel('Signature Recovery Rate')
    plt.title('Impact of Noise on Digital Signature Recovery with Reed-Solomon Code')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

