# socionics/data_processing.py

import json
import os
import logging
from datetime import datetime

FUNCTIONS = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]


def save_feedback(user_id, username, statement, corrected_correlations, positive_feedback,
                 feedback_data_file='data/feedback_data.jsonl',
                 user_statements_file='data/user_db.json'):
    """
    Сохраняет обратную связь пользователя, включая новое утверждение и корреляции.

    Args:
        user_id (int): Telegram ID пользователя.
        username (str): Имя пользователя.
        statement (str): Утверждение пользователя.
        corrected_correlations (dict): Корреляции функций.
        positive_feedback (bool): Флаг положительной обратной связи.
        feedback_data_file (str, optional): Путь к файлу обратной связи. Defaults to 'data/feedback_data.jsonl'.
        user_statements_file (str, optional): Путь к файлу пользовательских утверждений. Defaults to 'data/user_db.json'.
    """
    # Преобразуем все значения корреляций в стандартные float
    corrected_correlations = {k: float(v) for k, v in corrected_correlations.items()}

    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "username": username,
        "statement": statement,
        "function_correlation": corrected_correlations,
        "positive_feedback": positive_feedback
    }
    try:
        # Сохраняем обратную связь в feedback_data_file
        os.makedirs(os.path.dirname(feedback_data_file), exist_ok=True)
        with open(feedback_data_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        logging.info(f"Обратная связь от пользователя {username} сохранена в {feedback_data_file}.")

        # Если это новое утверждение, сохраняем его в user_statements_file
        if not positive_feedback:
            user_statements = []
            if os.path.exists(user_statements_file):
                with open(user_statements_file, 'r', encoding='utf-8') as f:
                    try:
                        user_statements = json.load(f)
                    except json.JSONDecodeError:
                        logging.error(f"Ошибка декодирования JSON в {user_statements_file}. Файл будет перезаписан.")
                        user_statements = []

            # Проверяем, есть ли уже такое утверждение
            if not any(entry['statement'].strip().lower() == statement.strip().lower() for entry in user_statements):
                user_statements.append({
                    "statement": statement,
                    "function_correlation": corrected_correlations
                })
                with open(user_statements_file, 'w', encoding='utf-8') as f:
                    json.dump(user_statements, f, ensure_ascii=False, indent=4)
                logging.info(f"Новое утверждение сохранено в {user_statements_file}.")
    except Exception as e:
        logging.error(f"Не удалось сохранить обратную связь: {e}")


def load_feedback_data(feedback_data_file='data/feedback_data.jsonl'):
    """
    Загружает данные обратной связи из файла.

    Args:
        feedback_data_file (str, optional): Путь к файлу обратной связи. Defaults to 'data/feedback_data.jsonl'.

    Returns:
        list: Список словарей с обратной связью.
    """
    feedback_data = []
    if os.path.exists(feedback_data_file):
        with open(feedback_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    feedback_entry = json.loads(line)
                    # Проверяем, что есть необходимые поля
                    if 'statement' in feedback_entry and 'function_correlation' in feedback_entry:
                        feedback_data.append({
                            'statement': feedback_entry['statement'],
                            'function_correlation': feedback_entry['function_correlation']
                        })
                except json.JSONDecodeError as e:
                    logging.error(f"Ошибка декодирования JSON: {e}")
    return feedback_data
