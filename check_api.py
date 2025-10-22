


import requests
import json

def check_api():

    base_url = "http://127.0.0.1:8000"
    
    print("🔍 Проверка работоспособности API...")
    
    try:

        response = requests.get(f"{base_url}/ping", timeout=5)
        if response.status_code == 200:
            print("✅ Ping: OK")
        else:
            print(f"❌ Ping: Ошибка {response.status_code}")
            return False
        

        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Главная страница: OK")
        else:
            print(f"❌ Главная страница: Ошибка {response.status_code}")
        

        test_text = "Test 123"
        print(f"🧪 Тестируем кодирование/декодирование: '{test_text}'")
        

        response = requests.post(
            f"{base_url}/encode",
            json={"text": test_text},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Encode: Ошибка {response.status_code}")
            return False
        
        wav_data = response.json()["data"]
        print(f"✅ Encode: OK (размер: {len(wav_data)} символов)")
        

        response = requests.post(
            f"{base_url}/decode",
            json={"data": wav_data},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Decode: Ошибка {response.status_code}")
            return False
        
        decoded_text = response.json()["text"]
        print(f"✅ Decode: OK (результат: '{decoded_text}')")
        

        if decoded_text == test_text:
            print("🎯 Тест полного цикла: УСПЕШНО!")
            return True
        else:
            print(f"⚠️  Тест полного цикла: Неточность (ожидали '{test_text}', получили '{decoded_text}')")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Не удается подключиться к серверу. Убедитесь, что сервер запущен на http://127.0.0.1:8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Таймаут запроса. Сервер может быть перегружен.")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Проверка API Text-to-Audio")
    print("=" * 40)
    
    success = check_api()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Все проверки пройдены! API готов к работе.")
        print("\nДоступные URL:")
        print("- Главная: http://127.0.0.1:8000")
        print("- Документация: http://127.0.0.1:8000/docs")
        print("- Ping: http://127.0.0.1:8000/ping")
    else:
        print("❌ Некоторые проверки не прошли.")
        print("\nПроверьте:")
        print("1. Запущен ли сервер: uvicorn app:app --host 127.0.0.1 --port 8000")
        print("2. Установлены ли зависимости: pip install -r requirements.txt")
        print("3. Не заблокирован ли порт 8000")
