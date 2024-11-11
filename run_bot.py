# run_bot.py
from bot.architecture import setup_bot
from neural_network.inference import predict_correlations
from socionics.calculations import calculate_traits, predict_socionics_types, get_agree_disagree_types
from socionics.data_processing import load_feedback_data
from config.settings import MODEL_PATH, SCALER_PATH, TALANOV_STATEMENTS_FILE, USER_STATEMENTS_FILE, FEEDBACK_DATA_FILE, FUNCTIONS, SOCIONICS_TYPES
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import joblib
import logging
import os

def main():
    # Настройка логирования (можно перенести в architecture.py)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/bot.logs"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Запуск бота...")

    # Инициализация модели эмбеддингов
    embedding_model = SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')
    logger.info("Модель эмбеддингов загружена.")

    # Проверка, существует ли сохранённая модель и скейлер
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        logger.info("Загрузка сохранённой модели и скейлера...")
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Модель и скейлер успешно загружены.")
    else:
        logger.info("Сохранённая модель или скейлер не найдены. Обучение модели с нуля...")
        from neural_network.training import train_and_save_model
        model, scaler = train_and_save_model(
            embedding_model=embedding_model,
            talanov_data_file=TALANOV_STATEMENTS_FILE,
            user_statements_file=USER_STATEMENTS_FILE,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            functions=FUNCTIONS
        )
        logger.info("Модель обучена и сохранена.")

    # Инициализация бота
    application = setup_bot()

    # Сохранение необходимых объектов в bot_data для доступа из других модулей
    application.bot_data['embedding_model'] = embedding_model
    application.bot_data['model'] = model
    application.bot_data['scaler'] = scaler

    # Запуск бота
    application.run_polling()

if __name__ == '__main__':
    main()
