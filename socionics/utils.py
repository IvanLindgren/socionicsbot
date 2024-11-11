# socionics/utils.py

import re
import logging

FUNCTIONS = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]


def parse_corrected_correlations(text):
    """
    Парсит исправленные корреляции, введённые пользователем.

    Args:
        text (str): Введённый пользователем текст корреляций.

    Returns:
        dict or None: Словарь с корреляциями или None, если парсинг не удался.
    """
    # Проверяем, является ли ввод упрощённым (начинается с + или -)
    simplified_pattern = r'^([+-]\w+(?:,\s*[+-]\w+)*)$'
    detailed_pattern = r'^(\w+):\s*(-?\d+\.?\d*)$'

    if re.match(simplified_pattern, text):
        # Обработка упрощённой обратной связи
        correlations = {}
        parts = text.split(',')
        for part in parts:
            part = part.strip()
            match = re.match(r'^([+-])(\w+)$', part)
            if match:
                sign, func = match.groups()
                if func not in FUNCTIONS:
                    logging.warning(f"Функция {func} не распознана.")
                    return None
                # Присваиваем веса на основе знака
                if sign == '+':
                    correlations[func] = 1.0  # Максимальное положительное влияние
                else:
                    correlations[func] = -1.0  # Максимальное отрицательное влияние
            else:
                logging.warning(f"Часть '{part}' не соответствует паттерну.")
                return None
        return correlations
    else:
        # Обработка детализированной обратной связи
        lines = text.strip().split('\n')
        corrected_correlations = {}
        for line in lines:
            match = re.match(r'^(\w+):\s*(-?\d+\.?\d*)$', line.strip())
            if match:
                func, value = match.groups()
                if func in FUNCTIONS:
                    corr_value = float(value)
                    # Ограничиваем значения диапазоном [-1, 1]
                    corrected_correlations[func] = max(-1.0, min(1.0, corr_value))
                else:
                    logging.warning(f"Функция {func} не распознана.")
                    return None
            else:
                logging.warning(f"Строка '{line}' не соответствует формату.")
                return None
        return corrected_correlations
