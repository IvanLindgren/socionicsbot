# bot/architecture.py

import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters, ContextTypes
)
from bot.handlers import (
    start,
    info_command,
    cancel_command,
    add_statement_start,
    add_statement_receive,
    add_correlations_receive,
    oprosnik_start,
    handle_oprosnik_answer,
    process_oprosnik_results,
    neurotype_start,
    neurotype_receive_description,
    process_neurotype_description,
    button_handler,
    error_handler,
    handle_general_text  # Импортируем новый обработчик
)
from bot.states import BotStates
from config.settings import TELEGRAM_BOT_TOKEN
from bot.commands import start_command, info_command, cancel_command
from bot.states import BotStates
from bot.utils import inline_buttons, main_menu_keyboard
from config.settings import LOGGING_LEVEL, LOGGING_FORMAT, TELEGRAM_BOT_TOKEN
from neural_network.model import create_multi_output_model
from neural_network.training import train_and_save_model
from neural_network.inference import predict_correlations
from socionics.data_processing import load_feedback_data
import os


def setup_bot():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Регистрация команд
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('info', info_command))
    application.add_handler(CommandHandler('cancel', cancel_command))

    # ConversationHandler для добавления утверждения
    add_conversation = ConversationHandler(
        entry_points=[CommandHandler('add', add_statement_start)],
        states={
            BotStates.WAITING_FOR_STATEMENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, add_statement_receive)
            ],
            BotStates.WAITING_FOR_CORRELATIONS_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, add_correlations_receive)
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel_command)]
    )
    application.add_handler(add_conversation)

    # ConversationHandler для опросника
    oprosnik_conversation = ConversationHandler(
        entry_points=[CommandHandler('oprosnik', oprosnik_start)],
        states={
            BotStates.OPROSNIK_PROCESSING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_oprosnik_answer)
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel_command)]
    )
    application.add_handler(oprosnik_conversation)

    # ConversationHandler для нейротипирования
    neurotype_conversation = ConversationHandler(
        entry_points=[CommandHandler('neurotype', neurotype_start)],
        states={
            BotStates.WAITING_FOR_NEUROTYPE_DESCRIPTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, neurotype_receive_description)
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel_command)]
    )
    application.add_handler(neurotype_conversation)

    # Обработчик инлайн-кнопок
    application.add_handler(CallbackQueryHandler(button_handler))

    # Обработчик общих текстовых сообщений
    general_text_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_general_text)
    application.add_handler(general_text_handler)

    # Обработчик ошибок
    application.add_error_handler(error_handler)

    return application


async def button_handler(update, context):
    """
    Обработчик нажатий на инлайн-кнопки.
    """
    query = update.callback_query
    await query.answer()

    if query.data == 'add_statement':
        await add_statement_start(update, context)
    elif query.data == 'oprosnik':
        await oprosnik_start(update, context)
    elif query.data == 'neurotype':
        await neurotype_start(update, context)
    elif query.data == 'info':
        await info_command(update, context)
    else:
        await query.edit_message_text(text="❓ Неизвестная команда.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """
    Глобальный обработчик ошибок.
    """
    logging.error(f"Update {update} caused error {context.error}")
    # Опционально: отправить сообщение пользователю о возникшей ошибке
    try:
        if isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❗️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
            )
    except Exception as e:
        logging.error(f"Не удалось отправить сообщение об ошибке пользователю: {e}")


async def default_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик сообщений по умолчанию, если они не попадают под другие категории.
    """
    await update.message.reply_text(
        "❓ Неизвестная команда или сообщение. Используйте /info для получения списка доступных команд.",
        reply_markup=main_menu_keyboard()
    )
