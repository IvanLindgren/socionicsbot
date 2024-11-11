import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# Параметры
MODEL_PATH = '../models/talanovCorrelationsPavlov.keras'
SCALER_PATH = '../models/label_scaler.pkl'
TEST_DATA_PATH = '../data/talanovstatements.json'  # Путь к тестовым данным
FUNCTIONS = ["ЧИ", "БИ", "ЧС", "БС", "БЛ", "ЧЛ", "БЭ", "ЧЭ", "БК", "ЧК", "БД", "ЧД"]

# Загрузка модели и скейлера
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Загрузка модели эмбеддингов
embedding_model = SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')

# Загрузка тестовых данных
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

statements = [entry['statement'] for entry in test_data]
actual_correlations = [entry['function_correlation'] for entry in test_data]


# Функция предсказания корреляций
def predict_correlations(statement):
    emb = embedding_model.encode([statement])
    predictions = model.predict(emb)
    prediction_dict = {}
    for i, func in enumerate(FUNCTIONS):
        prediction_dict[func] = predictions[i][0][0]  # Предполагается, что модель возвращает список
    # Обратное масштабирование
    prediction_array = np.array([prediction_dict[func] for func in FUNCTIONS]).reshape(1, -1)
    prediction_scaled = scaler.inverse_transform(prediction_array)[0]
    # Ограничение значений в диапазоне [-1, 1]
    correlations = {func: max(-1.0, min(1.0, prediction_scaled[i])) for i, func in enumerate(FUNCTIONS)}
    return correlations


# Вычисление ошибок и сбор данных по функциям
errors = []
function_errors = {func: [] for func in FUNCTIONS}

for actual, statement in zip(actual_correlations, statements):
    predicted = predict_correlations(statement)
    # Вычисление модуля корреляций
    actual_mod = {k: abs(v) for k, v in actual.items()}
    predicted_mod = {k: abs(v) for k, v in predicted.items()}
    # Вычисление ошибки как разницы модулей
    error = sum(abs(actual_mod.get(func, 0) - predicted_mod.get(func, 0)) for func in FUNCTIONS)
    errors.append(error)
    # Сбор ошибок по функциям
    for func in FUNCTIONS:
        actual_val = actual_mod.get(func, 0)
        predicted_val = predicted_mod.get(func, 0)
        func_error = abs(actual_val - predicted_val)
        function_errors[func].append(func_error)

# Накопительные суммы ошибок (если необходимо)
cumulative_errors = np.cumsum(errors)
num_statements = np.arange(1, len(errors) + 1)


# Функция для адаптивных меток оси X
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


# Получаем позиции меток оси X
x_ticks = get_x_ticks(len(errors))

# Построение графика ошибки по каждому утверждению
plt.figure(figsize=(12, 6))
plt.plot(num_statements, errors, marker='o', linestyle='-', color='blue')
plt.title('Ошибка предсказания модели по каждому утверждению')
plt.xlabel('Номер утверждения')
plt.ylabel('Ошибка предсказания')
plt.xticks(x_ticks)  # Устанавливаем адаптивные метки
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_error_individual.png')  # Сохранение графика
plt.show()


# Функция оценки предсказательной точности
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


# Расчёт точности
total_error = sum(errors)
max_error_per_statement = len(FUNCTIONS) * 2  # Максимальная ошибка на одно утверждение
max_possible_error = len(statements) * max_error_per_statement
accuracy = calculate_accuracy(total_error, max_possible_error)
print(f"Предсказательная точность модели: {accuracy:.2f}%")

# Вычисление средней ошибки для каждой функции
average_errors = {func: np.mean(errs) for func, errs in function_errors.items()}

# Сортировка функций по средней ошибке
sorted_functions = sorted(average_errors.items(), key=lambda item: item[1])

# Определение лучших и худших функций
best_functions = sorted_functions[:3]  # Три функции с наименьшей ошибкой
worst_functions = sorted_functions[-3:]  # Три функции с наибольшей ошибкой

# Вывод результатов
print("\nЛучшие предсказываемые функции:")
for func, error in best_functions:
    print(f"{func}: Средняя ошибка = {error:.4f}")

print("\nХудшие предсказываемые функции:")
for func, error in reversed(worst_functions):
    print(f"{func}: Средняя ошибка = {error:.4f}")
