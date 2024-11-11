# bot/utils.py

from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton


def main_menu_keyboard():
    """
    Создает клавиатуру с основными командами бота.

    Returns:
        ReplyKeyboardMarkup: Клавиатура для основного меню.
    """
    keyboard = [
        ['/add', '/oprosnik'],
        ['/neurotype', '/info'],
        ['/update_model', '/cancel']
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def confirmation_keyboard():
    """
    Создает клавиатуру для подтверждения (Да/Нет).

    Returns:
        ReplyKeyboardMarkup: Клавиатура с кнопками "Да" и "Нет".
    """
    keyboard = [['Да', 'Нет']]
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)


def inline_buttons():
    """
    Создает инлайн-кнопки для быстрого доступа к командам.

    Returns:
        InlineKeyboardMarkup: Инлайн-клавиатура с кнопками.
    """
    keyboard = [
        [InlineKeyboardButton("Добавить Утверждение", callback_data='add_statement')],
        [InlineKeyboardButton("Пройти Опросник", callback_data='oprosnik')],
        [InlineKeyboardButton("Нейротипирование", callback_data='neurotype')],
        [InlineKeyboardButton("Информация", callback_data='info')]
    ]
    return InlineKeyboardMarkup(keyboard)
