# bot/handlers.py

import logging
import os
import json
import random
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters
)
from bot.states import BotStates
from bot.utils import main_menu_keyboard, confirmation_keyboard, inline_buttons
from neural_network.inference import predict_correlations
from socionics.calculations import (
    calculate_traits,
    predict_socionics_types,
    get_agree_disagree_types,
    FUNCTIONS,
    modify_coefficients_based_on_answer
)
from socionics.utils import parse_corrected_correlations
from socionics.data_processing import save_feedback
from config.settings import SOCIONICS_TYPES, TALANOV_STATEMENTS_FILE, USER_STATEMENTS_FILE, FEEDBACK_DATA_FILE, \
    DEVELOPER_CHAT_ID


# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name

    await update.message.reply_text(
        "👋 Привет! Я соционический бот. Я могу анализировать ваши утверждения, помогать определить социотип и многое другое.\n\n"
        "Используйте кнопки ниже или команды для взаимодействия со мной.",
        reply_markup=inline_buttons()
    )
    logging.info(f"Пользователь {username} (ID: {user_id}) начал взаимодействие с ботом.")


# bot/handlers.py

# ... существующий код ...

from telegram.ext import filters


# Обработчик общих текстовых сообщений для предсказания
async def handle_general_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name
    text = update.message.text.strip()

    if not text:
        await update.message.reply_text(
            "❗️ Пожалуйста, введите текст для анализа или используйте команды бота.",
            reply_markup=main_menu_keyboard()
        )
        logging.warning(f"Пользователь {username} (ID: {user_id}) отправил пустое сообщение.")
        return

    logging.info(f"Пользователь {username} (ID: {user_id}) отправил утверждение для анализа: {text}")

    # Предсказание корреляций
    correlations = predict_correlations(
        statement=text,
        embedding_model=context.bot_data['embedding_model'],
        model=context.bot_data['model'],
        scaler=context.bot_data['scaler'],
        talanov_data_file=TALANOV_STATEMENTS_FILE,
        user_data_file=FEEDBACK_DATA_FILE,
        user_statements_file=USER_STATEMENTS_FILE
    )

    if not correlations:
        await update.message.reply_text(
            "❗️ Не удалось получить корреляции. Пожалуйста, попробуйте позже.",
            reply_markup=main_menu_keyboard()
        )
        logging.error(f"Не удалось получить корреляции для утверждения от пользователя {username} (ID: {user_id}).")
        return

    # Вычисляем признаки на основе корреляций
    traits = calculate_traits(correlations)

    # Нейротипирование
    probabilities = predict_socionics_types(traits, SOCIONICS_TYPES)

    # Определение типов, которые согласились и не согласились бы
    agree_disagree = get_agree_disagree_types(probabilities)

    # Формирование ответа
    reply_text = "📊 *Результаты анализа утверждения*:\n\n"
    for type_name, prob in probabilities.items():
        reply_text += f"{type_name}: {prob:.2f}%\n"

    reply_text += f"\n👍 *Положительные типы*: {', '.join(agree_disagree['agree'])}\n"
    reply_text += f"👎 *Отрицательные типы*: {', '.join(agree_disagree['disagree'])}\n"

    await update.message.reply_text(reply_text, parse_mode='Markdown', reply_markup=main_menu_keyboard())

    logging.info(f"Пользователь {username} (ID: {user_id}) получил результаты анализа.")


# Обработчик команды /info
async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    logging.info("Пользователь запросил информацию о боте.")


# Обработчик команды /cancel
async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name

    context.user_data.clear()
    await update.message.reply_text("❌ Действие отменено. Вы можете начать сначала.", reply_markup=main_menu_keyboard())
    logging.info(f"Пользователь {username} (ID: {user_id}) отменил текущий процесс.")


# Обработчик начала добавления утверждения (/add)
async def add_statement_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name

    await update.message.reply_text(
        "✏️ Пожалуйста, введите ваше утверждение, которое вы хотите добавить.",
        reply_markup=ReplyKeyboardMarkup([['/cancel']], resize_keyboard=True, one_time_keyboard=True)
    )
    logging.info(f"Пользователь {username} (ID: {user_id}) начал добавление нового утверждения.")
    return BotStates.WAITING_FOR_STATEMENT


# Обработчик получения утверждения (/add)
async def add_statement_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name
    statement = update.message.text.strip()

    if not statement:
        await update.message.reply_text("❗️ Утверждение не может быть пустым. Пожалуйста, введите ваше утверждение.")
        logging.warning(f"Пользователь {username} (ID: {user_id}) отправил пустое утверждение.")
        return BotStates.WAITING_FOR_STATEMENT

    # Сохраняем утверждение в user_data
    context.user_data['new_statement'] = statement
    await update.message.reply_text(
        "Спасибо! Теперь введите корреляции для вашего утверждения.\n\n"
        "Вы можете сделать это двумя способами:\n"
        "1. **Упрощённый формат**:\n"
        "`+БС, +БИ, -ЧС, -ЧЛ`\n"
        "2. **Детализированный формат**:\n"
        "`ЧИ: -0.07`\n"
        "`БИ: 0.9`\n"
        "Значения должны быть в диапазоне от -1 до 1.\n\n"
        "Пожалуйста, введите корреляции:",
        parse_mode='Markdown',
        reply_markup=ReplyKeyboardMarkup([['/cancel']], resize_keyboard=True, one_time_keyboard=True)
    )
    logging.info(f"Пользователь {username} (ID: {user_id}) ввёл утверждение: {statement}")
    return BotStates.WAITING_FOR_CORRELATIONS_INPUT


# Обработчик получения корреляций (/add)
async def add_correlations_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name
    text = update.message.text.strip()

    # Парсинг корреляций
    corrected_correlations = parse_corrected_correlations(text)
    if corrected_correlations:
        statement = context.user_data.get('new_statement')
        if not statement:
            await update.message.reply_text("❗️ Произошла ошибка. Пожалуйста, начните процесс заново.")
            logging.error(f"Пользователь {username} (ID: {user_id}) не предоставил утверждение после корреляций.")
            return ConversationHandler.END

        # Сохраняем обратную связь как отрицательную (пользователь предлагает коррекции)
        save_feedback(
            user_id=user_id,
            username=username,
            statement=statement,
            corrected_correlations=corrected_correlations,
            positive_feedback=False
        )

        # Отправляем корреляции разработчику (опционально)
        # await send_correlations_to_developer(context.bot, user_id, username, statement, corrected_correlations)

        await update.message.reply_text(
            "✅ Спасибо! Ваше утверждение и корреляции сохранены и будут рассмотрены разработчиком.",
            reply_markup=main_menu_keyboard()
        )

        logging.info(f"Пользователь {username} (ID: {user_id}) добавил новое утверждение с корреляциями.")
        return ConversationHandler.END
    else:
        await update.message.reply_text(
            "❗️ Пожалуйста, введите корректные корреляции в указанном формате.\n\n"
            "1. **Упрощённый формат**:\n"
            "`+БС, +БИ, -ЧС, -ЧЛ`\n"
            "2. **Детализированный формат**:\n"
            "`ЧИ: -0.07`\n"
            "`БИ: 0.9`\n"
            "Значения должны быть в диапазоне от -1 до 1.",
            parse_mode='Markdown',
            reply_markup=ReplyKeyboardMarkup([['/cancel']], resize_keyboard=True, one_time_keyboard=True)
        )
        logging.warning(f"Пользователь {username} (ID: {user_id}) ввёл некорректные корреляции: {text}")
        return BotStates.WAITING_FOR_CORRELATIONS_INPUT


# Обработчик команды /oprosnik (опросник)
async def oprosnik_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name

    # Загрузка утверждений
    talanov_data_file = TALANOV_STATEMENTS_FILE
    user_statements_file = USER_STATEMENTS_FILE

    talanov_statements = []
    if os.path.exists(talanov_data_file):
        with open(talanov_data_file, 'r', encoding='utf-8') as f:
            talanov_data = json.load(f)
            talanov_statements.extend([entry['statement'] for entry in talanov_data])

    user_statements = []
    if os.path.exists(user_statements_file):
        with open(user_statements_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
            user_statements.extend([entry['statement'] for entry in user_data])

    all_statements = talanov_statements + user_statements
    total_statements = len(all_statements)

    if total_statements == 0:
        await update.message.reply_text("Извините, нет доступных вопросов для опросника.")
        logging.warning(f"Пользователь {username} (ID: {user_id}) попытался пройти опросник без доступных вопросов.")
        return ConversationHandler.END

    num_questions = 10  # Можно сделать настраиваемым через config/settings.py
    if num_questions > total_statements:
        num_questions = total_statements

    # Выбираем случайные утверждения
    random_statements = random.sample(all_statements, num_questions)

    # Сохраняем состояние пользователя
    context.user_data['oprosnik'] = {
        'statements': random_statements,
        'answers': [],
        'current_question': 0
    }

    await update.message.reply_text(
        "📋 *Начинаем опросник.*\nПожалуйста, отвечайте цифрой от 1 до 5, где:\n"
        "1️⃣ - Совершенно не согласен\n"
        "2️⃣ - Скорее не согласен\n"
        "3️⃣ - Не знаю\n"
        "4️⃣ - Скорее согласен\n"
        "5️⃣ - Полностью согласен",
        parse_mode='Markdown',
        reply_markup=main_menu_keyboard()
    )

    await send_next_oprosnik_question(update, context)
    return BotStates.OPROSNIK_PROCESSING


# Функция для отправки следующего вопроса в опроснике
async def send_next_oprosnik_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    oprosnik_data = context.user_data.get('oprosnik')
    if not oprosnik_data:
        await update.message.reply_text("❗️ Произошла ошибка. Пожалуйста, начните опросник заново.")
        logging.error("Данные опросника отсутствуют в user_data.")
        return ConversationHandler.END

    current_question = oprosnik_data['current_question']
    statements = oprosnik_data['statements']

    if current_question < len(statements):
        question = statements[current_question]
        await update.message.reply_text(
            f"❓ *Вопрос {current_question + 1} из {len(statements)}*:\n\n{question}",
            parse_mode='Markdown'
        )
        oprosnik_data['current_question'] += 1
        return BotStates.OPROSNIK_PROCESSING
    else:
        # Опросник завершен
        await process_oprosnik_results(update, context)
        return ConversationHandler.END


# Обработчик ответа на вопрос опросника
async def handle_oprosnik_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    oprosnik_data = context.user_data.get('oprosnik')
    if not oprosnik_data:
        await update.message.reply_text("❗️ Произошла ошибка. Пожалуйста, начните опросник заново.")
        logging.error("Данные опросника отсутствуют в user_data.")
        return ConversationHandler.END

    answer = update.message.text.strip()
    if answer not in ['1', '2', '3', '4', '5']:
        await update.message.reply_text("❗️ Пожалуйста, отвечайте цифрой от 1 до 5.")
        logging.warning(f"Пользователь {update.effective_user.username} ввёл некорректный ответ: {answer}")
        return BotStates.OPROSNIK_PROCESSING

    oprosnik_data['answers'].append(int(answer))
    logging.info(f"Пользователь {update.effective_user.username} ответил: {answer}")
    return await send_next_oprosnik_question(update, context)


# Обработка результатов опросника
async def process_oprosnik_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    oprosnik_data = context.user_data.get('oprosnik')
    if not oprosnik_data:
        await update.message.reply_text("❗️ Произошла ошибка при обработке результатов опросника.")
        logging.error("Данные опросника отсутствуют при обработке результатов.")
        return

    statements = oprosnik_data['statements']
    answers = oprosnik_data['answers']

    if len(statements) != len(answers):
        await update.message.reply_text("❗️ Количество ответов не совпадает с количеством вопросов.")
        logging.error("Количество ответов не совпадает с количеством вопросов.")
        return

    # Инициализируем словарь для накопления коэффициентов функций
    accumulated_correlations = {func: 0.0 for func in FUNCTIONS}

    for statement, answer in zip(statements, answers):
        # Получаем исходные корреляции для утверждения
        correlations = predict_correlations(
            statement=statement,
            embedding_model=context.bot_data['embedding_model'],
            model=context.bot_data['model'],
            scaler=context.bot_data['scaler'],
            talanov_data_file=TALANOV_STATEMENTS_FILE,
            user_data_file=FEEDBACK_DATA_FILE,
            user_statements_file=USER_STATEMENTS_FILE
        )

        if not correlations:
            logging.warning(f"Не удалось получить корреляции для утверждения: {statement}")
            continue

        # Модифицируем коэффициенты на основе ответа пользователя
        modified_correlations = modify_coefficients_based_on_answer(correlations, answer)

        if modified_correlations is None:
            # Ответ 3, коэффициенты игнорируются
            continue

        # Накопление коэффициентов
        for func in FUNCTIONS:
            accumulated_correlations[func] += modified_correlations.get(func, 0.0)

    # Нормализация коэффициентов
    num_questions = len(statements)
    for func in accumulated_correlations:
        accumulated_correlations[func] /= num_questions

    # Вычисляем признаки на основе накопленных коэффициентов
    traits = calculate_traits(accumulated_correlations)

    # Нейротипирование
    probabilities = predict_socionics_types(traits, SOCIONICS_TYPES)

    # Определение типов, которые согласились и не согласились бы
    agree_disagree = get_agree_disagree_types(probabilities)

    # Формирование ответа
    reply_text = "📊 *Результаты опросника*:\n\n"
    for type_name, prob in probabilities.items():
        reply_text += f"{type_name}: {prob:.2f}%\n"

    reply_text += f"\n👍 *Положительные типы*: {', '.join(agree_disagree['agree'])}\n"
    reply_text += f"👎 *Отрицательные типы*: {', '.join(agree_disagree['disagree'])}\n"

    await update.message.reply_text(reply_text, parse_mode='Markdown', reply_markup=main_menu_keyboard())

    # Сохранение обратной связи (опционально)
    # for statement, answer in zip(statements, answers):
    #     correlations = predict_correlations(...)
    #     save_feedback(user_id, username, statement, correlations, positive_feedback=True)

    logging.info(f"Пользователь {update.effective_user.username} завершил опросник и получил результаты.")


# Обработчик команды /neurotype
async def neurotype_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name

    # Извлекаем описание из команды, если оно было предоставлено
    user_input = update.message.text[len('/neurotype'):].strip()
    if user_input:
        description = user_input
        await process_neurotype_description(update, context, description)
        return ConversationHandler.END
    else:
        await update.message.reply_text(
            "🧠 Пожалуйста, предоставьте описание для нейротипирования.\n\n"
            "Пример:\n"
            "/neurotype Я люблю помогать другим и стремлюсь к гармонии.",
            reply_markup=ReplyKeyboardMarkup([['/cancel']], resize_keyboard=True, one_time_keyboard=True)
        )
        logging.info(f"Пользователь {username} (ID: {user_id}) начал процесс нейротипирования.")
        return BotStates.WAITING_FOR_NEUROTYPE_DESCRIPTION


# Обработчик ввода описания для нейротипирования
async def neurotype_receive_description(update: Update, context: ContextTypes.DEFAULT_TYPE):
    description = update.message.text.strip()
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name

    if not description:
        await update.message.reply_text(
            "❗️ Описание не может быть пустым. Пожалуйста, предоставьте описание для нейротипирования.",
            reply_markup=ReplyKeyboardMarkup([['/cancel']], resize_keyboard=True, one_time_keyboard=True)
        )
        logging.warning(f"Пользователь {username} (ID: {user_id}) отправил пустое описание для нейротипирования.")
        return BotStates.WAITING_FOR_NEUROTYPE_DESCRIPTION

    await process_neurotype_description(update, context, description)
    return ConversationHandler.END


# Функция для обработки описания и предсказания нейротипа
async def process_neurotype_description(update: Update, context: ContextTypes.DEFAULT_TYPE, description: str):
    user = update.effective_user
    user_id = user.id
    username = user.username if user.username else user.first_name

    # Предсказание корреляций
    correlations = predict_correlations(
        statement=description,
        embedding_model=context.bot_data['embedding_model'],
        model=context.bot_data['model'],
        scaler=context.bot_data['scaler'],
        talanov_data_file=TALANOV_STATEMENTS_FILE,
        user_data_file=FEEDBACK_DATA_FILE,
        user_statements_file=USER_STATEMENTS_FILE
    )

    if not correlations:
        await update.message.reply_text(
            "❗️ Не удалось получить корреляции. Пожалуйста, попробуйте позже.",
            reply_markup=main_menu_keyboard()
        )
        logging.error(f"Не удалось получить корреляции для описания от пользователя {username} (ID: {user_id}).")
        return

    # Вычисляем признаки на основе корреляций
    traits = calculate_traits(correlations)

    # Нейротипирование
    probabilities = predict_socionics_types(traits, SOCIONICS_TYPES)

    # Определение типов, которые согласились и не согласились бы
    agree_disagree = get_agree_disagree_types(probabilities)

    # Формирование ответа
    reply_text = "📊 *Результаты нейротипирования*:\n\n"
    for type_name, prob in probabilities.items():
        reply_text += f"{type_name}: {prob:.2f}%\n"

    reply_text += f"\n👍 *Положительные типы*: {', '.join(agree_disagree['agree'])}\n"
    reply_text += f"👎 *Отрицательные типы*: {', '.join(agree_disagree['disagree'])}\n"

    await update.message.reply_text(reply_text, parse_mode='Markdown', reply_markup=main_menu_keyboard())

    # Сохранение обратной связи (опционально)
    # save_feedback(user_id, username, description, correlations, positive_feedback=True)

    logging.info(f"Пользователь {username} (ID: {user_id}) завершил нейротипирование и получил результаты.")


# Обработчик нажатий на инлайн-кнопки
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        logging.warning(f"Пользователь нажал неизвестную инлайн-кнопку: {query.data}")


# Обработчик ошибок
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Update {update} вызвал ошибку {context.error}")
    # Опционально: отправить сообщение пользователю о возникшей ошибке
    try:
        if isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❗️ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
            )
    except Exception as e:
        logging.error(f"Не удалось отправить сообщение об ошибке пользователю: {e}")

# Функция для отправки корреляций разработчику
async def send_correlations_to_developer(bot, user_id, username, statement, correlations):
#Отправляет корреляции разработчику с информацией о пользователе.
    message = (
       f"🔄 *Новые корреляции от пользователя*:\n\n"
       f"👤 Ник: @{username}\n"
        f"🆔 Telegram ID: {user_id}\n\n"
        f"📝 *Утверждение*:\n{statement}\n\n"
         f"📊 *Корреляции*:\n"
    )
    for func, corr in correlations.items():
         message += f"{func}: {corr:.4f}\n"

    try:
        await bot.send_message(chat_id=DEVELOPER_CHAT_ID, text=message, parse_mode='Markdown')
        logging.info(f"Сообщение разработчику от пользователя {username} (ID: {user_id}) отправлено.")
    except Exception as e:
        logging.error(f"Не удалось отправить сообщение разработчику: {e}")
