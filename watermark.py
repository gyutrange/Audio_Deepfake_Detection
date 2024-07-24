from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
import time
import random

# RSA 키 생성
key = RSA.generate(2048)
private_key = key.export_key()
public_key = key.publickey().export_key()

with open("private.pem", "wb") as f:
    f.write(private_key)

with open("public.pem", "wb") as f:
    f.write(public_key)

def create_watermark(data, private_key_file):
    # 타임스탬프 추가
    timestamp = int(time.time())
    watermark_data = f"{data}|{timestamp}"

    # 해시 생성
    hash_obj = SHA256.new(watermark_data.encode('utf-8'))

    # RSA 서명
    with open(private_key_file, "rb") as f:
        private_key = RSA.import_key(f.read())
    signature = pkcs1_15.new(private_key).sign(hash_obj)

    # 해시 값을 사용하여 16비트로 줄이기
    short_watermark_bits = bin(int(signature.hex(), 16))[-16:]
    
    return short_watermark_bits

# 워터마크 생성 예시 (random값과 키 비교)
watermark_bits = create_watermark(random.random(), "private.pem")
print(f"Watermark Bits: {watermark_bits}")

# 워터마크 이진 데이터 저장
with open("watermark_bits.txt", "w") as f:
    f.write(watermark_bits)

def verify_watermark(watermark_data, signature, public_key_file):
    # 해시 생성
    hash_obj = SHA256.new(watermark_data.encode('utf-8'))

    # RSA 서명 검증
    with open(public_key_file, "rb") as f:
        public_key = RSA.import_key(f.read())

    try:
        pkcs1_15.new(public_key).verify(hash_obj, signature)
        print("The watermark is authentic.")
    except (ValueError, TypeError):
        print("The watermark is not authentic.")

# 필요한 경우, 검증 함수 호출을 위한 추가 코드

