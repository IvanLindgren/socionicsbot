# config/settings.py

import os
import json
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env (если используется)
load_dotenv()

# Токен Telegram-бота
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'ваш_токен_здесь')

# Chat ID разработчика для получения обратной связи
DEVELOPER_CHAT_ID = int(os.getenv('DEVELOPER_CHAT_ID', 'ваш_chat_id_здесь'))

# Пути к файлам данных
MODEL_PATH = os.getenv('MODEL_PATH', 'models/talanovCorrelations.keras')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/label_scaler.pkl')
USER_STATEMENTS_FILE = os.getenv('USER_STATEMENTS_FILE', 'data/user_db.json')
FEEDBACK_DATA_FILE = os.getenv('FEEDBACK_DATA_FILE', 'data/feedback_data.jsonl')
TALANOV_STATEMENTS_FILE = os.getenv('TALANOV_STATEMENTS_FILE', 'data/talanovstatements.json')
SOCIONICS_TYPES_FILE = os.getenv('SOCIONICS_TYPES_FILE', 'data/socionic_types.json')

# Настройки логирования
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO')
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Параметры модели
FUNCTIONS = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]

# Загрузка соционических типов из socionic_types.json
if os.path.exists(SOCIONICS_TYPES_FILE):
    with open(SOCIONICS_TYPES_FILE, 'r', encoding='utf-8') as f:
        SOCIONICS_TYPES = json.load(f)
else:
    SOCIONICS_TYPES = {}
    print(f"Файл {SOCIONICS_TYPES_FILE} не найден. SOCIONICS_TYPES пуст.")
