# bot/commands.py

from telegram import Update
from telegram.ext import ContextTypes
from bot.states import BotStates
from bot.utils import main_menu_keyboard, inline_buttons
from neural_network.inference import predict_correlations
from socionics.calculations import calculate_traits, predict_socionics_types, get_agree_disagree_types
from socionics.data_processing import save_feedback
import logging

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды /start. Приветствует пользователя и показывает главное меню.
    """
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Привет, {user.first_name}! Я соционический бот, готовый помочь вам с анализом утверждений и определением социотипа.",
        reply_markup=main_menu_keyboard()
    )

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды /info. Предоставляет информацию о боте и доступных командах.
    """
    info_text = (
        "ℹ️ *Информация о Боте*:\n\n"
        "Я соционический бот, который может:\n"
        "🔹 Анализировать ваши утверждения и предоставлять корреляции с соционическими функциями.\n"
        "🔹 Позволять добавлять новые утверждения для улучшения модели.\n"
        "🔹 Предоставлять опросники для определения вашего социотипа.\n"
        "🔹 Выполнять нейротипирование на основе вашего описания.\n\n"
        "*Доступные команды*:\n"
        "🔹 /start - Начать взаимодействие с ботом.\n"
        "🔹 /add - Добавить новое утверждение в базу данных.\n"
        "🔹 /oprosnik - Пройти опросник для определения социотипа.\n"
        "🔹 /neurotype - Провести нейротипирование по вашему описанию.\n"
        "🔹 /update_model - Обновить модель на основе новой обратной связи.\n"
        "🔹 /info - Показать информацию о боте и доступных командах.\n"
        "🔹 /cancel - Отменить текущий процесс."
    )
    await update.message.reply_text(info_text, parse_mode='Markdown', reply_markup=main_menu_keyboard())

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды /cancel. Отменяет текущий процесс и возвращает в главное меню.
    """
    user = update.effective_user
    context.user_data.clear()
    await update.message.reply_text(
        "❌ Действие отменено. Вы можете начать сначала.",
        reply_markup=main_menu_keyboard()
    )
    logging.info(f"Пользователь {user.username} отменил текущий процесс.")

# Добавьте другие команды здесь, такие как /add, /oprosnik, /neurotype и т.д.
