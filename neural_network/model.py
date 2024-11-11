# neural_network/model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
import logging

# Сопоставление имен функций на русском и английском (ASCII)
FUNCTION_NAME_MAPPING = {
    "ЧИ": "ChI",
    "БИ": "BI",
    "ЧС": "ChS",
    "БС": "BS",
    "БЛ": "BL",
    "ЧЛ": "ChL",
    "БЭ": "BE",
    "ЧЭ": "ChE",
    "БК": "BK",
    "ЧК": "ChK",
    "БД": "BD",
    "ЧД": "ChD"
}


def create_multi_output_model(input_dim, output_funcs):
    """
    Создаёт и компилирует многовыходную модель нейронной сети.

    Args:
        input_dim (int): Размерность входного вектора (эмбеддингов).
        output_funcs (list): Список функций, для которых будут предсказываться корреляции.

    Returns:
        tensorflow.keras.Model: Скомпилированная модель.
    """
    input_layer = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = []
    for func in output_funcs:
        ascii_func_name = FUNCTION_NAME_MAPPING.get(func, func)
        outputs.append(Dense(1, activation='linear', name=ascii_func_name)(x))

    model = Model(inputs=input_layer, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mean_absolute_error'] * len(outputs)  # Повторяем метрику для каждого выхода
    )

    logging.info("Модель успешно создана и скомпилирована.")
    return model
