
"""FastAPI сервис, кодирующий текст в WAV и декодирующий его обратно.

✦ Реализуйте функции `text_to_audio` и `audio_to_text`.
✦ Формат аудио: 44100Hz, 16‑bit PCM, mono.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import wave
import io
import base64
from pathlib import Path

class EncodeRequest(BaseModel):
    text: str

class EncodeResponse(BaseModel):
    data: str

class DecodeRequest(BaseModel):
    data: str

class DecodeResponse(BaseModel):
    text: str

app = FastAPI()

SAMPLE_RATE = 44_100
CHANNELS = 1
BIT_DEPTH = 16

def text_to_audio(text: str) -> bytes:
    
 base_freq = 1000  # Базовая частота в Гц
    freq_step = 150   # Шаг частоты между символами
    symbol_duration = 0.1  # Длительность одного символа в секундах
    chars = "0123456789abcdefghijklmnopqrstuvwxyz "
    char_to_freq = {char: base_freq + i * freq_step for i, char in enumerate(chars)}
    message_data = []
    samples_per_symbol = int(SAMPLE_RATE * symbol_duration)
    text_to_encode = text

    for char in text_to_encode:
        char_lower = char.lower()
        if char_lower in char_to_freq:
            message_data.append(char_to_freq[char_lower])
        else:
            message_data.append(char_to_freq[' '])  # Неизвестный символ заменяем на пробел
    total_samples = len(message_data) * samples_per_symbol
    audio_signal = np.zeros(total_samples, dtype=np.float32)
    for i, freq in enumerate(message_data):
        start_sample = i * samples_per_symbol
        end_sample = start_sample + samples_per_symbol
        t = np.linspace(0, symbol_duration, samples_per_symbol, endpoint=False)
        signal = 0.8 * np.sin(2 * np.pi * freq * t)
        fade_samples = min(220, samples_per_symbol // 20)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            signal[:fade_samples] *= fade_in
            fade_out = np.linspace(1, 0, fade_samples)
            signal[-fade_samples:] *= fade_out

        audio_signal[start_sample:end_sample] = signal
    audio_int16 = (audio_signal * 32767 * 0.9).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BIT_DEPTH // 8)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

  return buf.getvalue()

def audio_to_text(wav_bytes: bytes) -> str:

    
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
        
        if len(audio_data) == 0:
            return ""
        audio_signal = audio_data.astype(np.float32) / 32767.0
        base_freq = 1000
        freq_step = 150
        symbol_duration = 0.1
        samples_per_symbol = int(SAMPLE_RATE * symbol_duration)

        if len(audio_signal) < samples_per_symbol:
            return ""
        chars = "0123456789abcdefghijklmnopqrstuvwxyz "
        freq_to_char = {base_freq + i * freq_step: char for i, char in enumerate(chars)}
        decoded_chars = []
        num_segments = len(audio_signal) // samples_per_symbol
        max_segments = min(num_segments, 50000)  # Максимум 50000 символов для безопасности
        
\
    for char in text:

        bits += format(ord(char), '08b')
    return bits

def bits_to_text(bits: str) -> str:

    text = ""

    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = bits[i:i+8]
            char_code = int(byte, 2)
            if 0 <= char_code <= 127:  # Только ASCII символы
                text += chr(char_code)
    return text

def calculate_checksum(bits: str) -> str:

    checksum = 0
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte_val = int(bits[i:i+8], 2)
            checksum ^= byte_val
    return format(checksum, '08b')

def generate_fsk_signal(bits: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:


    freq_0 = 1200  # Hz для бита 0
    freq_1 = 2400  # Hz для бита 1
    

    bit_duration = 0.01  # 10ms на бит
    samples_per_bit = int(sample_rate * bit_duration)
    

    preamble = "01010101" * 4  # 32 бита преамбулы
    

    checksum = calculate_checksum(bits)
    

    data_length = format(len(bits), '016b')  # 16 бит для длины
    full_bits = preamble + data_length + bits + checksum
    

    redundant_bits = ""
    for bit in full_bits:
        redundant_bits += bit * 3
    
    signal = np.array([], dtype=np.float32)
    
    for bit_char in redundant_bits:
        bit = int(bit_char)
        freq = freq_1 if bit else freq_0
        

        t = np.linspace(0, bit_duration, samples_per_bit, False)
        bit_signal = np.sin(2 * np.pi * freq * t)
        

        ramp_len = min(samples_per_bit // 10, 50)
        if ramp_len > 0:
            bit_signal[:ramp_len] *= np.linspace(0, 1, ramp_len)
            bit_signal[-ramp_len:] *= np.linspace(1, 0, ramp_len)
        
        signal = np.concatenate([signal, bit_signal])
    


    

    freq_0 = 1200
    freq_1 = 2400
    bit_duration = 0.01
    samples_per_bit = int(sample_rate * bit_duration)
    

    preamble_pattern = "01010101" * 4
    redundant_preamble = ""
    for bit in preamble_pattern:
        redundant_preamble += bit * 3
    

    decoded_bits = ""
    num_bits = len(audio) // samples_per_bit
    
    for i in range(num_bits):
        start_idx = i * samples_per_bit
        end_idx = start_idx + samples_per_bit
        
        if end_idx > len(audio):
            break
            
        bit_audio = audio[start_idx:end_idx]
        

        t = np.linspace(0, bit_duration, samples_per_bit, False)
        signal_0 = np.sin(2 * np.pi * freq_0 * t)
        signal_1 = np.sin(2 * np.pi * freq_1 * t)
        

        corr_0 = np.abs(np.mean(bit_audio * signal_0))
        corr_1 = np.abs(np.mean(bit_audio * signal_1))
        

        decoded_bits += "1" if corr_1 > corr_0 else "0"
    

    preamble_pos = decoded_bits.find(redundant_preamble)
    if preamble_pos == -1:
        return ""
    

    data_start = preamble_pos + len(redundant_preamble)
    if data_start >= len(decoded_bits):
        return ""
    

    length_bits_redundant = decoded_bits[data_start:data_start + 48]
    if len(length_bits_redundant) < 48:
        return ""
    

    length_bits = ""
    for i in range(0, 48, 3):
        bit_group = length_bits_redundant[i:i+3]

        length_bits += "1" if bit_group.count("1") > bit_group.count("0") else "0"
    
    try:
        data_length = int(length_bits, 2)
    except ValueError:
        return ""
    

    data_start_redundant = data_start + 48
    data_bits_needed = data_length * 3  # с избыточностью
    checksum_bits_needed = 8 * 3  # контрольная сумма с избыточностью
    
    total_needed = data_start_redundant + data_bits_needed + checksum_bits_needed
    if total_needed > len(decoded_bits):
        return ""
    

    data_redundant = decoded_bits[data_start_redundant:data_start_redundant + data_bits_needed]
    checksum_redundant = decoded_bits[data_start_redundant + data_bits_needed:data_start_redundant + data_bits_needed + checksum_bits_needed]
    

    data_bits = ""
    for i in range(0, len(data_redundant), 3):
        if i + 3 <= len(data_redundant):
            bit_group = data_redundant[i:i+3]
            data_bits += "1" if bit_group.count("1") > bit_group.count("0") else "0"
    
    checksum_bits = ""
    for i in range(0, len(checksum_redundant), 3):
        if i + 3 <= len(checksum_redundant):
            bit_group = checksum_redundant[i:i+3]
            checksum_bits += "1" if bit_group.count("1") > bit_group.count("0") else "0"
    

    expected_checksum = calculate_checksum(data_bits)
    if checksum_bits != expected_checksum:

        pass
    

    return bits_to_text(data_bits)

def text_to_audio(text: str) -> bytes:

    if not text:
        return _empty_wav(0.1)
    

    bits = text_to_bits(text)
    

    signal = generate_fsk_signal(bits)
    

    duration = len(signal) / SAMPLE_RATE
    if duration > 10.0:

        max_chars = int(len(text) * 10.0 / duration)
        text = text[:max_chars]
        bits = text_to_bits(text)
        signal = generate_fsk_signal(bits)
    

    signal_int16 = (signal * 32767).astype(np.int16)
    

            

        audio = audio_int16.astype(np.float32) / 32768.0
        

        return decode_fsk_signal(audio)
        
    except Exception:
        return ""





@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = Path(file).parent / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding='utf-8'))
    else:
        return HTMLResponse(content="<h1>Web interface not found</h1>")

@app.post("/encode", response_model=EncodeResponse)
async def encode_text(request: EncodeRequest):
    from audio_encoder import text_to_audio
    wav_bytes = text_to_audio(request.text)
    wav_base64 = base64.b64encode(wav_bytes).decode("utf-8")
    return EncodeResponse(data=wav_base64)

@app.post("/decode", response_model=DecodeResponse)
async def decode_audio(request: DecodeRequest):
    from audio_decoder import audio_to_text
    wav_bytes = base64.b64decode(request.data)
    text = audio_to_text(wav_bytes)
    return DecodeResponse(text=text)

@app.get("/ping")
async def ping():
    return "ok"
