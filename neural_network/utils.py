# neural_network/utils.py

import logging

FUNCTIONS = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]


def validate_correlation_values(correlations):
    """
    Проверяет, что все значения корреляций находятся в диапазоне [-1, 1].

    Args:
        correlations (dict): Словарь с корреляциями функций.

    Returns:
        bool: True, если все значения в диапазоне, иначе False.
    """
    for func, value in correlations.items():
        if not (-1.0 <= value <= 1.0):
            logging.warning(f"Значение корреляции для функции {func} ({value}) выходит за диапазон [-1, 1].")
            return False
    return True


def preprocess_statement(statement):
    """
    Предобрабатывает утверждение перед генерацией эмбеддингов.

    Args:
        statement (str): Входное утверждение.

    Returns:
        str: Предобработанное утверждение.
    """
    # Пример предобработки: удаление лишних пробелов и приведение к нижнему регистру
    return statement.strip().lower()


def postprocess_predictions(predictions, scaler, functions):
    """
    Постобрабатывает предсказания модели, включая обратное масштабирование и ограничение значений.

    Args:
        predictions (numpy.ndarray): Предсказания модели.
        scaler (MinMaxScaler): Скейлер для обратного преобразования.
        functions (list): Список функций.

    Returns:
        dict: Словарь с корреляциями функций.
    """
    # Обратное масштабирование
    prediction_scaled = scaler.inverse_transform(predictions)

    # Формирование корреляций
    correlations = {func: max(-1.0, min(1.0, prediction_scaled[0][i])) for i, func in enumerate(functions)}

    return correlations
