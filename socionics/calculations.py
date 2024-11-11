# socionics/calculations.py

import logging

FUNCTIONS = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]

def calculate_traits(correlations):
    """
    Вычисляет соционические признаки на основе корреляций функций.

    Args:
        correlations (dict): Словарь с корреляциями функций.

    Returns:
        dict: Словарь с вычисленными признаками.
    """
    traits = {}

    # Квестимность и Деклатимность
    questim_funcs = ["БК", "ЧК"]
    declatim_funcs = ["БД", "ЧД"]
    questimity = sum(correlations.get(func, 0) for func in questim_funcs) - sum(
        correlations.get(func, 0) for func in declatim_funcs)
    traits['Квестимность'] = questimity

    # Интуиция и Сенсорика
    intuitive_funcs = ["БИ", "ЧИ"]
    sensing_funcs = ["БС", "ЧС"]
    intuition = sum(correlations.get(func, 0) for func in intuitive_funcs) - sum(
        correlations.get(func, 0) for func in sensing_funcs)
    traits['Интуиция'] = intuition

    # Демократизм и Аристократизм
    democratic_funcs = ["БК", "ЧД"]
    aristocratic_funcs = ["ЧК", "БД"]
    democratism = sum(correlations.get(func, 0) for func in democratic_funcs) - sum(
        correlations.get(func, 0) for func in aristocratic_funcs)
    traits['Демократизм'] = democratism

    # Веселость и Серьезность
    merry_funcs = ["ЧЭ", "БЛ"]
    serious_funcs = ["БЭ", "ЧЛ"]
    merriness = sum(correlations.get(func, 0) for func in merry_funcs) - sum(
        correlations.get(func, 0) for func in serious_funcs)
    traits['Веселость'] = merriness

    # Логика и Этика
    logical_funcs = ["БЛ", "ЧЛ"]
    ethical_funcs = ["ЧЭ", "БЭ"]
    logic = sum(correlations.get(func, 0) for func in logical_funcs) - sum(
        correlations.get(func, 0) for func in ethical_funcs)
    traits['Логика'] = logic

    # Экстраверсия и Интроверсия
    extraverted_funcs = [func for func in correlations if func.startswith('Ч')]
    introverted_funcs = [func for func in correlations if func.startswith('Б')]
    extraversion = sum(correlations.get(func, 0) for func in extraverted_funcs) - sum(
        correlations.get(func, 0) for func in introverted_funcs)
    traits['Экстраверсия'] = extraversion

    # Иррациональность и Рациональность
    irrational_funcs = ["ЧИ", "БИ", "БС", "ЧС"]
    rational_funcs = ["ЧЭ", "БЭ", "БЛ", "ЧЛ"]
    irrationality = sum(correlations.get(func, 0) for func in irrational_funcs) - sum(
        correlations.get(func, 0) for func in rational_funcs)
    traits['Иррациональность'] = irrationality

    # Рассудительность и Решительность
    judicious_funcs = ["ЧИ", "БС"]
    decisive_funcs = ["ЧС", "БИ"]
    judiciousness = sum(correlations.get(func, 0) for func in judicious_funcs) - sum(
        correlations.get(func, 0) for func in decisive_funcs)
    traits['Рассудительность'] = judiciousness

    # Статика и Динамика
    static_funcs = ["БЛ", "БЭ", "ЧИ", "ЧС"]
    dynamic_funcs = ["ЧЛ", "ЧЭ", "БИ", "БС"]
    statics = sum(correlations.get(func, 0) for func in static_funcs) - sum(
        correlations.get(func, 0) for func in dynamic_funcs)
    traits['Статика'] = statics

    return traits


def predict_socionics_types(traits, socionics_types):
    """
    Предсказывает вероятности социотипов на основе вычисленных признаков.

    Args:
        traits (dict): Словарь с вычисленными признаками.
        socionics_types (dict): Словарь соционических типов и их характеристик.

    Returns:
        dict: Словарь с вероятностями социотипов.
    """
    type_scores = {}

    for type_name, characteristics in socionics_types.items():
        score = 0
        for trait, alignment in characteristics.items():
            trait_value = traits.get(trait, 0)
            score += trait_value * alignment
        type_scores[type_name] = score

    # Убираем отрицательные баллы
    for type_name in type_scores:
        if type_scores[type_name] < 0:
            type_scores[type_name] = 0

    total_score = sum(type_scores.values())
    if total_score == 0:
        probabilities = {type_name: 0 for type_name in type_scores}
    else:
        probabilities = {type_name: (score / total_score) * 100 for type_name, score in type_scores.items()}

    # Сортировка типов по вероятности (от большего к меньшему)
    sorted_probabilities = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return dict(sorted_probabilities)


def get_agree_disagree_types(probabilities, top_n=3, bottom_n=3):
    """
    Определяет согласные и несогласные социотипы на основе вероятностей.

    Args:
        probabilities (dict): Словарь с вероятностями социотипов.
        top_n (int, optional): Количество верхних типов. Defaults to 3.
        bottom_n (int, optional): Количество нижних типов. Defaults to 3.

    Returns:
        dict: Словарь с ключами 'agree' и 'disagree' и списками соответствующих типов.
    """
    sorted_types = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_agree = [type_name for type_name, prob in sorted_types[:top_n]]
    top_disagree = [type_name for type_name, prob in sorted_types[-bottom_n:]]
    return {
        "agree": top_agree,
        "disagree": top_disagree
    }


def modify_coefficients_based_on_answer(correlations, answer):
    """
    Модифицирует коэффициенты на основе ответа пользователя в опроснике.

    Args:
        correlations (dict): Исходные корреляции функций.
        answer (int): Ответ пользователя (1-5).

    Returns:
        dict or None: Модифицированные корреляции или None, если ответ игнорирует коэффициенты.
    """
    # Создаем копию корреляций, чтобы не изменять исходные данные
    modified_correlations = correlations.copy()

    if answer == 1:
        # Умножаем все коэффициенты на -1
        for func in modified_correlations:
            modified_correlations[func] *= -1
    elif answer == 2:
        # Умножаем на -1 и делим на 2
        for func in modified_correlations:
            modified_correlations[func] = (modified_correlations[func] * -1) / 2
    elif answer == 3:
        # Игнорируем коэффициенты
        return None
    elif answer == 4:
        # Делим на 2
        for func in modified_correlations:
            modified_correlations[func] /= 2
    elif answer == 5:
        # Коэффициенты остаются без изменений
        pass
    else:
        # На случай некорректного ответа
        logging.warning(f"Неверный ответ пользователя: {answer}")
        return None

    return modified_correlations
