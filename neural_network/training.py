# neural_network/training.py

import json
import numpy as np
import os
import logging
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .model import create_multi_output_model
from socionics.data_processing import load_feedback_data

FUNCTIONS = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]


def train_and_save_model(embedding_model, talanov_data_file, user_statements_file, model_path, scaler_path, functions):
    """
    Обучает и сохраняет многовыходную модель нейронной сети.

    Args:
        embedding_model (SentenceTransformer): Модель для генерации эмбеддингов.
        talanov_data_file (str): Путь к файлу с утверждениями Таланова.
        user_statements_file (str): Путь к файлу с пользовательскими утверждениями.
        model_path (str): Путь для сохранения обученной модели.
        scaler_path (str): Путь для сохранения скейлера.
        functions (list): Список функций для предсказания.

    Returns:
        tensorflow.keras.Model: Обученная модель.
        MinMaxScaler: Обученный скейлер.
    """
    try:
        # Загрузка данных Таланова
        with open(talanov_data_file, 'r', encoding='utf-8') as f:
            talanov_data = json.load(f)
        logging.info(f"Загружено {len(talanov_data)} утверждений из {talanov_data_file}.")

        # Загрузка пользовательских утверждений
        if os.path.exists(user_statements_file):
            with open(user_statements_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            logging.info(f"Загружено {len(user_data)} пользовательских утверждений из {user_statements_file}.")
        else:
            user_data = []
            logging.info(f"Файл {user_statements_file} не найден. Продолжаем без пользовательских утверждений.")

        # Объединение данных
        combined_data = talanov_data + user_data
        logging.info(f"Объединено {len(combined_data)} утверждений для обучения.")

        if len(combined_data) < 10:
            logging.error(
                f"Недостаточно данных для обучения модели. Требуется минимум 10 образцов, получено {len(combined_data)}.")
            raise ValueError("Недостаточно данных для обучения модели.")

        # Подготовка данных
        statements = [entry['statement'] for entry in combined_data]
        correlations = [entry['function_correlation'] for entry in combined_data]

        # Генерация эмбеддингов
        embeddings = embedding_model.encode(statements, show_progress_bar=True)
        logging.info("Эмбеддинги успешно сгенерированы.")

        # Преобразование корреляций в массивы
        labels = np.array([[corr.get(func, 0.0) for func in functions] for corr in correlations])

        # Масштабирование меток в диапазон [-1, 1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        labels_scaled = scaler.fit_transform(labels)
        logging.info("Метки успешно масштабированы.")

        # Сохранение скейлера
        scaler_dir = os.path.dirname(scaler_path)
        if scaler_dir and not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Скейлер сохранён в {scaler_path}.")

        # Создание модели
        input_dim = embeddings.shape[1]
        model = create_multi_output_model(input_dim, functions)
        logging.info("Модель создана.")

        # Разделение данных на обучающую и валидационную выборки
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels_scaled,
            test_size=0.2,
            random_state=42
        )
        logging.info(
            f"Данные разделены на обучающую ({len(X_train)} samples) и валидационную ({len(X_val)} samples) выборки.")

        # Преобразование меток для многовыходной модели
        y_train_list = [y_train[:, i].reshape(-1, 1) for i in range(len(functions))]
        y_val_list = [y_val[:, i].reshape(-1, 1) for i in range(len(functions))]

        # Обучение модели
        logging.info("Начало обучения модели...")
        history = model.fit(
            X_train, y_train_list,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val_list),
            verbose=1
        )
        logging.info("Обучение модели завершено.")

        # Сохранение модели
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(model_path)
        logging.info(f"Модель сохранена в {model_path}.")

        return model, scaler
    except Exception as e:
        print("Ошбика: " + str(e))

    def retrain_model(embedding_model, model, scaler, talanov_data_file, user_statements_file, feedback_data_file,
                      model_path, scaler_path, functions, epochs=10, batch_size=32):
        """
        Переобучает многовыходную модель нейронной сети на основе обратной связи.

        Args:
            embedding_model (SentenceTransformer): Модель для генерации эмбеддингов.
            model (tensorflow.keras.Model): Загруженная модель.
            scaler (MinMaxScaler): Загруженный скейлер.
            talanov_data_file (str): Путь к файлу с утверждениями Таланова.
            user_statements_file (str): Путь к файлу с пользовательскими утверждениями.
            feedback_data_file (str): Путь к файлу с обратной связью пользователей.
            model_path (str): Путь для сохранения обновленной модели.
            scaler_path (str): Путь для сохранения обновленного скейлера.
            functions (list): Список функций для предсказания.
            epochs (int, optional): Количество эпох для переобучения. Defaults to 10.
            batch_size (int, optional): Размер батча. Defaults to 32.

        Returns:
            tensorflow.keras.Model: Обновленная модель.
            MinMaxScaler: Обновленный скейлер.
        """

        try:
            # Загрузка исходных данных
            with open(talanov_data_file, 'r', encoding='utf-8') as f:
                talanov_data = json.load(f)
            logging.info(f"Загружено {len(talanov_data)} утверждений из {talanov_data_file}.")

            # Загрузка пользовательских утверждений
            if os.path.exists(user_statements_file):
                with open(user_statements_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                logging.info(f"Загружено {len(user_data)} пользовательских утверждений из {user_statements_file}.")
            else:
                user_data = []
                logging.info(f"Файл {user_statements_file} не найден. Продолжаем без пользовательских утверждений.")

            # Загрузка данных обратной связи
            feedback_data = load_feedback_data(feedback_data_file)
            logging.info(f"Загружено {len(feedback_data)} записей обратной связи из {feedback_data_file}.")

            # Объединение данных
            combined_data = talanov_data + user_data + feedback_data
            logging.info(f"Объединено {len(combined_data)} утверждений для переобучения.")

            # Проверка количества данных
            n_samples = len(combined_data)
            if n_samples < 10:
                logging.error(
                    f"Недостаточно данных для переобучения модели. Требуется минимум 10 образцов, получено {n_samples}.")
                raise ValueError("Недостаточно данных для переобучения модели.")

            # Подготовка данных
            statements = [entry['statement'] for entry in combined_data]
            correlations = [entry.get('function_correlation', {}) for entry in combined_data]

            # Генерация эмбеддингов
            embeddings = embedding_model.encode(statements, show_progress_bar=True)
            logging.info("Эмбеддинги успешно сгенерированы.")

            # Преобразование корреляций в массивы
            labels = np.array([[corr.get(func, 0.0) for func in functions] for corr in correlations])

            # Масштабирование меток в диапазон [-1, 1]
            labels_scaled = scaler.fit_transform(labels)
            logging.info("Метки успешно масштабированы.")

            # Сохранение скейлера
            joblib.dump(scaler, scaler_path)
            logging.info(f"Скейлер сохранён в {scaler_path}.")

            # Разделение данных на обучающую и валидационную выборки
            X_train, X_val, y_train, y_val = train_test_split(
                embeddings, labels_scaled,
                test_size=0.2,
                random_state=42
            )
            logging.info(
                f"Данные разделены на обучающую ({len(X_train)} samples) и валидационную ({len(X_val)} samples) выборки.")

            # Преобразование меток для многовыходной модели
            y_train_list = [y_train[:, i].reshape(-1, 1) for i in range(len(functions))]
            y_val_list = [y_val[:, i].reshape(-1, 1) for i in range(len(functions))]

            # Обучение модели
            logging.info("Начало переобучения модели...")
            history = model.fit(
                X_train, y_train_list,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val_list),
                verbose=1
            )
            logging.info("Переобучение модели завершено.")

            # Сохранение модели
            model.save(model_path)
            logging.info(f"Модель сохранена в {model_path}.")

            # Очистка временного файла обратной связи после обучения
            open(feedback_data_file, 'w').close()
            logging.info(f"Временный файл обратной связи {feedback_data_file} очищен после переобучения.")

            return model, scaler
        except Exception as e:
            logging.error(f"Ошибка при переобучении модели: {e}")
            raise e
