# neural_network/inference.py

import json
import numpy as np
import os
import logging
from socionics.data_processing import load_feedback_data
from .utils import preprocess_statement, postprocess_predictions

FUNCTIONS = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]


def predict_correlations(statement, embedding_model, model, scaler, talanov_data_file, user_data_file,
                         user_statements_file):
    """
    Предсказывает корреляции соционических функций для заданного утверждения.

    Args:
        statement (str): Утверждение для анализа.
        embedding_model (SentenceTransformer): Модель для генерации эмбеддингов.
        model (tensorflow.keras.Model): Загруженная модель нейронной сети.
        scaler (MinMaxScaler): Скейлер для обратного преобразования предсказаний.
        talanov_data_file (str): Путь к файлу с утверждениями Таланова.
        user_data_file (str): Путь к файлу с обратной связью пользователей.
        user_statements_file (str): Путь к файлу с пользовательскими утверждениями.

    Returns:
        dict: Словарь с корреляциями функций.
    """
    statement_clean = statement.strip().lower()

    # Проверка в пользовательских утверждениях
    if os.path.exists(user_statements_file):
        with open(user_statements_file, 'r', encoding='utf-8') as f:
            try:
                user_statements = json.load(f)
                for entry in user_statements:
                    if entry['statement'].strip().lower() == statement_clean:
                        logging.info("Утверждение найдено в пользовательских данных.")
                        return entry['function_correlation']
            except json.JSONDecodeError:
                logging.error(f"Ошибка декодирования JSON в {user_statements_file}.")

    # Проверка в обратной связи
    feedback_data = load_feedback_data(user_data_file)
    for entry in feedback_data:
        if entry['statement'].strip().lower() == statement_clean:
            logging.info("Утверждение найдено в обратной связи.")
            return entry.get('function_correlation', entry.get('correlations', {}))

    # Если не найдено, предсказываем
    emb = embedding_model.encode([statement])
    predictions = model.predict(emb)

    # Формируем словарь предсказаний
    prediction_dict = {}
    for i, func in enumerate(FUNCTIONS):
        prediction_dict[func] = predictions[i][0][0]  # Предполагается, что predictions[i] имеет форму (1,1)

    # Обратное масштабирование и ограничение значений
    prediction_array = np.array([prediction_dict[func] for func in FUNCTIONS]).reshape(1, -1)
    prediction_scaled = scaler.inverse_transform(prediction_array)[0]

    correlations = {func: max(-1.0, min(1.0, prediction_scaled[i])) for i, func in enumerate(FUNCTIONS)}

    logging.info(f"Корреляции предсказаны для утверждения: {statement}")

    return correlations
