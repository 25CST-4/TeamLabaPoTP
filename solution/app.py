
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

SAMPLE_RATE = 44100
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
        
        for i in range(max_segments):
            start_sample = i * samples_per_symbol
            end_sample = start_sample + samples_per_symbol
            
            if end_sample > len(audio_signal):
                break
                
            segment = audio_signal[start_sample:end_sample]
            
            if len(segment) < samples_per_symbol:
                break
            segment_energy = np.mean(segment**2)
            if segment_energy < 0.01:
                decoded_chars.append(' ')
                continue
            window = np.hanning(len(segment))
            windowed_segment = segment * window
            fft = np.fft.fft(windowed_segment)
            freqs = np.fft.fftfreq(len(windowed_segment), 1/SAMPLE_RATE)
            magnitude = np.abs(fft)
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            if len(positive_magnitude) > 0:
                freq_mask = (positive_freqs >= 800) & (positive_freqs <= 6500)
                if np.any(freq_mask):
                    masked_magnitude = positive_magnitude[freq_mask]
                    masked_freqs = positive_freqs[freq_mask]
                    if len(masked_magnitude) > 0:
                        peak_idx = np.argmax(masked_magnitude)
                        detected_freq = abs(masked_freqs[peak_idx])
                        best_char = None
                        min_diff = float('inf')
                        
                        for known_freq, char in freq_to_char.items():
                            diff = abs(detected_freq - known_freq)
                            if diff < min_diff and diff < 80:
                                min_diff = diff
                                best_char = char
                        
                        if best_char is not None:
                            decoded_chars.append(best_char)
                        else:
                            decoded_chars.append(' ')
                    else:
                        decoded_chars.append(' ')
                else:
                    decoded_chars.append(' ')
            else:
                decoded_chars.append(' ')
        
        return ''.join(decoded_chars)
        
    except Exception as e:
        return ""

@app.get("/", response_class=HTMLResponse)
async def get_index():

    index_file = Path(__file__).parent / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding='utf-8'))
    else:
        return HTMLResponse(content="<h1>Web interface not found</h1>")

@app.post("/encode", response_model=EncodeResponse)
async def encode_text(request: EncodeRequest):
    wav_bytes = text_to_audio(request.text)
    wav_base64 = base64.b64encode(wav_bytes).decode("utf-8")
    return EncodeResponse(data=wav_base64)

@app.post("/decode", response_model=DecodeResponse)
async def decode_audio(request: DecodeRequest):
    wav_bytes = base64.b64decode(request.data)
    text = audio_to_text(wav_bytes)
    return DecodeResponse(text=text)

@app.get("/ping")
async def ping():
    return "ok"
