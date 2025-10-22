```bash
cd solution
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

```bash
curl http://127.0.0.1:8000/ping
curl -X POST "http://127.0.0.1:8000/encode" -H "Content-Type: application/json" -d '{"text": "Test"}'
```

- Главная: http://127.0.0.1:8000
- API документация: http://127.0.0.1:8000/docs
- Ping: http://127.0.0.1:8000/ping

```bash
netstat -an | findstr 8000


Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process

pip uninstall -r requirements.txt -y && pip install -r requirements.txt
```

```python
python -c "from app import text_to_audio, audio_to_text; t='Hello'; r=audio_to_text(text_to_audio(t)); print(f'Input: {t}, Output: {r}, Success: {t==r}')"
```
