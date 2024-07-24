
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import scipy.io.wavfile as wav
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
    timestamp = int(time.time())
    watermark_data = f"{data}|{timestamp}"

    hash_obj = SHA256.new(watermark_data.encode('utf-8'))

    with open(private_key_file, "rb") as f:
        private_key = RSA.import_key(f.read())
    signature = pkcs1_15.new(private_key).sign(hash_obj)

    watermark = watermark_data + "|" + signature.hex()
    watermark_bits = ''.join([bin(ord(c)).lstrip('0b').rjust(8, '0') for c in watermark])

    return watermark_bits

def load_watermark_bits(watermark_file):
    with open(watermark_file, 'r') as f:
        watermark_bits = f.read()
    return watermark_bits

def embed_watermark_spread_spectrum(audio_file, output_file, watermark_bits):
    rate, audio = wav.read(audio_file)
    audio = audio.astype(np.float32)

    watermark_bits = np.array(list(map(int, watermark_bits)))

    np.random.seed(0)
    spread_sequence = np.random.choice([1, -1], size=(len(watermark_bits), audio.shape[0]))

    for i, bit in enumerate(watermark_bits):
        if bit == 1:
            audio += spread_sequence[i]

    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    wav.write(output_file, rate, audio)

def process_files(directory):
    watermark_bits = create_watermark(random.random(), "private.pem")

    with open("watermark_bits.txt", "w") as f:
        f.write(watermark_bits)

    watermark_bits = load_watermark_bits('watermark_bits.txt')

    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            input_file = os.path.join(directory, file_name)
            output_file = os.path.join(directory, 'watermarked_' + file_name)
            embed_watermark_spread_spectrum(input_file, output_file, watermark_bits)

    messagebox.showinfo("Success", "Watermarking completed for all WAV files in the selected directory.")

def select_directory():
    directory = filedialog.askdirectory()
    if directory:
        process_files(directory)

# GUI 설정
root = tk.Tk()
root.title("Audio Watermarking")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

label = tk.Label(frame, text="Select a directory containing WAV files to watermark:")
label.pack(pady=5)

select_button = tk.Button(frame, text="Select Directory", command=select_directory)
select_button.pack(pady=5)

root.mainloop()

