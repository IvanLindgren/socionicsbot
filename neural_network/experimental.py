# experimentals.py

import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

# Параметры
MODEL_NAME = "deepvk/USER-bge-m3"
TRAIN_DATA_PATH = "../data/talanovstatements.json"  # Путь к обучающей выборке
TEST_DATA_PATH = "../data/talanovtestingstatements.json"    # Путь к тестовой выборке
SCALER_PATH = "../models/new_label_scaler.pkl"  # Путь к скейлеру
OUTPUT_PLOT_TRAIN = "experimentals_train_error_plot.png"  # Имя файла для графика обучения
OUTPUT_PLOT_TEST = "experimentals_test_error_plot.png"  # Имя файла для графика тестирования

FUNCTIONS = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]


def load_data(data_path):
    """Загружает данные из JSON файла."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    statements = [entry['statement'] for entry in data]
    correlations = [entry['function_correlation'] for entry in data]
    return statements, correlations


def correlations_to_array(correlations, functions):
    """
    Преобразует список словарей корреляций в двумерный numpy массив.

    :param correlations: Список словарей с корреляциями.
    :param functions: Список функций в порядке, в котором нужно извлекать значения.
    :return: 2D numpy массив с корреляциями.
    """
    return np.array([[entry.get(func, 0.0) for func in functions] for entry in correlations])


def encode_statements(model, statements):
    """Преобразует утверждения в эмбеддинги с помощью модели."""
    embeddings = model.encode(statements, normalize_embeddings=True)
    return embeddings


def train_regressor(X_train, y_train):
    """Обучает многовыходную регрессионную модель."""
    base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    multi_regressor = MultiOutputRegressor(base_regressor)
    multi_regressor.fit(X_train, y_train)
    return multi_regressor


def calculate_accuracy(total_error, max_possible_error):
    """
    Рассчитывает предсказательную точность модели в процентах.

    :param total_error: Суммарная ошибка модели.
    :param max_possible_error: Максимально возможная ошибка.
    :return: Точность в процентах.
    """
    if max_possible_error == 0:
        return 100.0
    accuracy = max(0.0, 100 - (total_error / max_possible_error) * 100)
    return accuracy


def get_x_ticks(num_statements_count):
    """
    Определяет позиции меток на оси X в зависимости от количества утверждений.

    :param num_statements_count: Общее количество утверждений
    :return: Список позиций для меток оси X
    """
    if num_statements_count <= 10:
        return list(range(1, num_statements_count + 1))
    elif num_statements_count <= 20:
        return list(range(1, num_statements_count + 1, 2))
    elif num_statements_count <= 30:
        return list(range(1, num_statements_count + 1, 3))
    else:
        step = max(1, num_statements_count // 10)
        return list(range(1, num_statements_count + 1, step))


def plot_errors(num_statements, errors, title, output_file):
    """Строит и сохраняет график ошибок."""
    x_ticks = get_x_ticks(len(num_statements))
    plt.figure(figsize=(12, 6))
    plt.plot(num_statements, errors, marker='o', linestyle='-', color='blue')
    plt.title(title)
    plt.xlabel('Номер утверждения')
    plt.ylabel('Ошибка предсказания')
    plt.xticks(x_ticks)  # Устанавливаем адаптивные метки
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)  # Сохранение графика
    plt.show()


def interactive_prediction(regressor, model, scaler):
    """Интерактивная функция для предсказания корреляций по пользовательскому утверждению."""
    print("\n=== Интерактивное Предсказание Корреляций ===")
    while True:
        user_input = input("\nВведите утверждение (или 'exit' для выхода): ")
        if user_input.lower() == 'exit':
            print("Выход из интерактивного режима.")
            break
        # Преобразование утверждения в эмбеддинг
        embedding = model.encode([user_input], normalize_embeddings=True)
        # Предсказание корреляций
        predicted_correlation_scaled = regressor.predict(embedding)[0]
        # Обратное масштабирование
        predicted_correlation = scaler.inverse_transform([predicted_correlation_scaled])[0]
        # Ограничение значений в диапазоне [-1, 1]
        predicted_correlation = np.clip(predicted_correlation, -1.0, 1.0)
        # Вывод результатов
        print("\nПредсказанные корреляции:")
        for func, value in zip(FUNCTIONS, predicted_correlation):
            print(f"{func}: {value:.4f}")


def main():
    # Загрузка обучающих данных
    print("Загрузка обучающих данных...")
    train_statements, train_correlations = load_data(TRAIN_DATA_PATH)
    train_correlations_array = correlations_to_array(train_correlations, FUNCTIONS)

    # Загрузка тестовых данных
    print("Загрузка тестовых данных...")
    test_statements, test_correlations = load_data(TEST_DATA_PATH)
    test_correlations_array = correlations_to_array(test_correlations, FUNCTIONS)

    # Загрузка модели эмбеддингов
    print(f"Загрузка модели эмбеддингов: {MODEL_NAME}...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    # Извлечение эмбеддингов для обучающих данных
    print("Извлечение эмбеддингов для обучающих данных...")
    train_embeddings = encode_statements(embedding_model, train_statements)

    # Извлечение эмбеддингов для тестовых данных
    print("Извлечение эмбеддингов для тестовых данных...")
    test_embeddings = encode_statements(embedding_model, test_statements)

    # Проверка наличия скейлера, если нет — обучение и сохранение
    try:
        print("Загрузка скейлера...")
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("Скейлер не найден. Обучение скейлера на обучающих данных...")
        scaler = StandardScaler()
        scaler.fit(train_correlations_array)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Скейлер обучен и сохранён по пути: {SCALER_PATH}")

    # Масштабирование меток
    print("Масштабирование меток обучающих данных...")
    train_correlations_scaled = scaler.transform(train_correlations_array)
    print("Масштабирование меток тестовых данных...")
    test_correlations_scaled = scaler.transform(test_correlations_array)

    # Обучение регрессионной модели
    print("Обучение регрессионной модели...")
    regressor = train_regressor(train_embeddings, train_correlations_scaled)

    # Предсказание на обучающих данных (для анализа)
    print("Предсказание на обучающих данных...")
    train_pred = regressor.predict(train_embeddings)
    train_errors = np.abs(np.abs(train_correlations_scaled) - np.abs(train_pred))
    total_train_error = np.sum(train_errors)
    max_error_per_statement = len(FUNCTIONS) * 2  # Максимальная ошибка на одно утверждение
    max_possible_train_error = len(train_statements) * max_error_per_statement
    train_accuracy = calculate_accuracy(total_train_error, max_possible_train_error)

    # Вычисление средней ошибки для каждой функции на обучающих данных
    print("Вычисление средней ошибки для каждой функции на обучающих данных...")
    average_train_errors = {}
    for i, func in enumerate(FUNCTIONS):
        average_train_errors[func] = np.mean(train_errors[:, i])

    # Сортировка функций по средней ошибке
    sorted_train_functions = sorted(average_train_errors.items(), key=lambda item: item[1])
    best_train_functions = sorted_train_functions[:3]  # Три функции с наименьшей ошибкой
    worst_train_functions = sorted_train_functions[-3:]  # Три функции с наибольшей ошибкой

    # Построение графика ошибки на обучающих данных
    print("Построение графика ошибки на обучающих данных...")
    individual_train_errors = np.sum(train_errors, axis=1)
    num_train_statements = np.arange(1, len(individual_train_errors) + 1)
    plot_errors(
        num_train_statements,
        individual_train_errors,
        'Ошибка предсказания модели на обучающих данных',
        OUTPUT_PLOT_TRAIN
    )

    # Предсказание на тестовых данных
    print("Предсказание на тестовых данных...")
    test_pred = regressor.predict(test_embeddings)
    test_errors = np.abs(np.abs(test_correlations_scaled) - np.abs(test_pred))
    total_test_error = np.sum(test_errors)
    max_possible_test_error = len(test_statements) * max_error_per_statement
    test_accuracy = calculate_accuracy(total_test_error, max_possible_test_error)

    # Вычисление средней ошибки для каждой функции на тестовых данных
    print("Вычисление средней ошибки для каждой функции на тестовых данных...")
    average_test_errors = {}
    for i, func in enumerate(FUNCTIONS):
        average_test_errors[func] = np.mean(test_errors[:, i])

    # Сортировка функций по средней ошибке
    sorted_test_functions = sorted(average_test_errors.items(), key=lambda item: item[1])
    best_test_functions = sorted_test_functions[:3]  # Три функции с наименьшей ошибкой
    worst_test_functions = sorted_test_functions[-3:]  # Три функции с наибольшей ошибкой

    # Построение графика ошибки на тестовых данных
    print("Построение графика ошибки на тестовых данных...")
    individual_test_errors = np.sum(test_errors, axis=1)
    num_test_statements = np.arange(1, len(individual_test_errors) + 1)
    plot_errors(
        num_test_statements,
        individual_test_errors,
        'Ошибка предсказания модели на тестовых данных',
        OUTPUT_PLOT_TEST
    )

    # Вывод результатов
    print(f"\nПредсказательная точность модели на обучающих данных: {train_accuracy:.2f}%")
    print("Лучшие предсказываемые функции на обучающих данных:")
    for func, error in best_train_functions:
        print(f"{func}: Средняя ошибка = {error:.4f}")

    print("\nХудшие предсказываемые функции на обучающих данных:")
    for func, error in reversed(worst_train_functions):
        print(f"{func}: Средняя ошибка = {error:.4f}")

    print(f"\nПредсказательная точность модели на тестовых данных: {test_accuracy:.2f}%")
    print("Лучшие предсказываемые функции на тестовых данных:")
    for func, error in best_test_functions:
        print(f"{func}: Средняя ошибка = {error:.4f}")

    print("\nХудшие предсказываемые функции на тестовых данных:")
    for func, error in reversed(worst_test_functions):
        print(f"{func}: Средняя ошибка = {error:.4f}")

    # Интерактивная функция для предсказания корреляций
    interactive_prediction(regressor, embedding_model, scaler)


if __name__ == "__main__":
    main()