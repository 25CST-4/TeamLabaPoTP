import uvicorn
import sys
import os
sys.path.append(os.getcwd())

if __name__ == "__main__":
    print("🚀 Запуск сервера на http://localhost:8000")
    print("📝 Для остановки нажмите Ctrl+C")
    try:
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        print("💡 Попробуйте запустить от имени администратора")
        print("💡 Или проверьте, не занят ли порт 8000")
