import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.src.layers import BatchNormalization, Dropout
from keras.src.optimizers import Adam
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

plt.style.use('fivethirtyeight')

# Загрузка данных
product = pd.read_csv('price_prediction_dataset.csv')
product.info()
print(product.describe())
print(product.head())

# Сохраняем исходные данные для будущего использования
original_product = product.copy()

# Удаляем колонку product, но сохраняем ее значения
product_names = product['product'].copy()
product_dates = product['date'].copy()

product = product.drop(['product'], axis=1)
y = product['price']
X = product.drop('price', axis=1)

# Подготовка признаков
x = pd.get_dummies(X, columns=['date', 'sales',
                               'discount', 'advertising',
                               'stock_level', 'month', 'day_of_week',
                               'is_weekend', 'day_of_year', 'quarter', 'product_encoded'], drop_first=True)
print(f"Размерность признаков после one-hot encoding: {x.shape}")


# Кастомное разделение на train/test согласно вашим требованиям
def custom_train_test_split(product, x, y, product_names, product_dates, product_dates_train):
    """
    Кастомное разделение данных:
    Тесты: строки 1169-1460, 2630-2921, 4091-4382
    Остальные - обучение
    """
    # Индексы для тестовых данных
    test_indices = list(range(1168, 1460)) + list(range(2629, 2921)) + list(range(4090, 4382))

    # Индексы для тренировочных данных
    train_indices = [i for i in range(len(product)) if i not in test_indices]

    print(f"Тренировочные данные: {len(train_indices)} строк")
    print(f"Тестовые данные: {len(test_indices)} строк")

    # Разделяем данные
    X_train = x.iloc[train_indices]
    X_test = x.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Сохраняем дополнительную информацию для тестовых данных
    test_info = {
        'indices': test_indices,
        'dates': product_dates.iloc[test_indices],
        'product_names': product_names.iloc[test_indices]
    }

    # Сохраняем дополнительную информацию для тестовых данных
    train_info = {
        'indices': train_indices,
        'dates': product_dates.iloc[train_indices],
        'product_names': product_names.iloc[train_indices]
    }

    return X_train, X_test, y_train, y_test, test_info, train_info

# Применяем кастомное разделение
X_train, X_test, y_train, y_test, test_info, train_info = custom_train_test_split(
    product, x, y, product_names, product_dates, product_dates
)


# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Построение модели
model = Sequential([
    # Input(shape=(X_train.shape[1],)),
    # Dense(16, activation='relu'),
    # Dense(16, activation='relu'),
    # Dense(16, activation='relu'),
    # Dense(8, activation='relu'),
    # Dense(1, activation='linear')
    # Input(shape=(X_train.shape[1],)),
    # Dense(128, activation='relu'),
    # Dense(64, activation='relu'),
    # Dense(64, activation='relu'),
    # Dense(32, activation='relu'),
    # Dense(1, activation='linear')  # для регрессии
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # для регрессии
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Обучение модели
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, callbacks=[early_stop], validation_split=0.2, verbose=1)

# УЛУЧШЕННАЯ ВИЗУАЛИЗАЦИЯ ФУНКЦИИ ПОТЕРЬ
plt.figure(figsize=(15, 5))

# График 1: Функция потерь (MSE)
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Функция потерь (MSE)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Добавляем аннотацию с финальными значениями потерь
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
plt.annotate(f'Final Train: {final_train_loss:.4f}',
             xy=(0.98, 0.95), xycoords='axes fraction',
             ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))
plt.annotate(f'Final Val: {final_val_loss:.4f}',
             xy=(0.98, 0.85), xycoords='axes fraction',
             ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", alpha=0.7))

# Дополнительный анализ сходимости
plt.figure(figsize=(12, 4))

# График 4: Отношение валидационной и тренировочной потерь
plt.subplot(1, 2, 1)
loss_ratio = np.array(history.history['val_loss']) / np.array(history.history['loss'])
plt.plot(loss_ratio, linewidth=2, color='purple')
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Ideal ratio = 1')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Val Loss / Train Loss', fontsize=12)
plt.title('Отношение валидационной и тренировочной потерь', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# График 5: Разница между валидационной и тренировочной потерей
plt.subplot(1, 2, 2)
loss_diff = np.array(history.history['val_loss']) - np.array(history.history['loss'])
plt.plot(loss_diff, linewidth=2, color='brown')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No overfitting')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Val Loss - Train Loss', fontsize=12)
plt.title('Разница потерь (переобучение анализ)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Оценка модели
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Предсказания
pred = model.predict(X_test_scaled)

# Предсказания
pred_train = model.predict(X_train_scaled)

# Создание DataFrame с предсказаниями и сохранение в файл
predictions_df = pd.DataFrame({
    "index": test_info['indices'],
    "date": test_info['dates'].values,
    "product": test_info['product_names'].values,
    "predicted_price": pred.flatten()
})

# Создание DataFrame с предсказаниями и сохранение в файл
predictions_train = pd.DataFrame({
    "index": train_info['indices'],
    "date": train_info['dates'].values,
    "product": train_info['product_names'].values,
    "predicted_price": pred_train.flatten()
})

# Сохраняем в CSV файл с разделителем ","
predictions_df.to_csv("price_predictions.csv", index=False, sep=',')

# Также сохраняем в текстовый файл в требуемом формате
with open("price_predictions.txt", "w", encoding='utf-8') as file:
    # Записываем заголовок
    file.write("index,date,product,predicted_price\n")

    # Записываем данные
    for i in range(len(predictions_df)):
        row = predictions_df.iloc[i]
        file.write(f"{row['index']},{row['date']},{row['product']},{row['predicted_price']:.4f}\n")

# Также сохраняем в текстовый файл в требуемом формате
with open("price_predictions_train.txt", "w", encoding='utf-8') as file:
    # Записываем заголовок
    file.write("index,date,product,predicted_price\n")

    # Записываем данные
    for i in range(len(predictions_train)):
        row = predictions_train.iloc[i]
        file.write(f"{row['index']},{row['date']},{row['product']},{row['predicted_price']:.4f}\n")

print(f"Предсказания сохранены в файлы:")
print(f"- price_predictions.csv ({len(predictions_df)} записей)")
print(f"- price_predictions.txt ({len(predictions_df)} записей)")
print(f"- price_predictions_train.txt ({len(predictions_train)} записей)")

# Создаем DataFrame для сравнения фактических и предсказанных цен
comparison_df = pd.DataFrame({
    "index": test_info['indices'],
    "date": test_info['dates'].values,
    "product": test_info['product_names'].values,
    "actual_price": y_test.values,
    "predicted_price": pred.flatten()
})

# Визуализация сравнения первых 100 предсказаний
plt.figure(figsize=(16, 8))

# Берем первые 100 записей для визуализации
n_show = min(100, len(comparison_df))
comparison_show = comparison_df.head(n_show)

# Создаем индекс для оси X
x_index = range(n_show)

plt.plot(x_index, comparison_show['actual_price'],
         label='Фактическая цена', linewidth=2, marker='o', markersize=4)
plt.plot(x_index, comparison_show['predicted_price'],
         label='Предсказанная цена', linewidth=2, marker='s', markersize=4)

plt.xlabel('Номер наблюдения', fontsize=14)
plt.ylabel('Цена', fontsize=14)
plt.title(f'Сравнение фактических и предсказанных цен (первые {n_show} наблюдений)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Добавляем значения ошибок в заголовок
plt.figtext(0.5, 0.01, f'Test MAE: {mae:.4f}, Test Loss: {loss:.4f}',
            ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

plt.tight_layout()
plt.show()

# Дополнительная визуализация: scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(comparison_df['actual_price'], comparison_df['predicted_price'], alpha=0.6)
plt.plot([comparison_df['actual_price'].min(), comparison_df['actual_price'].max()],
         [comparison_df['actual_price'].min(), comparison_df['actual_price'].max()],
         'r--', linewidth=2, label='Идеальные предсказания')

plt.xlabel('Фактическая цена', fontsize=14)
plt.ylabel('Предсказанная цена', fontsize=14)
plt.title('Фактические vs Предсказанные цены (все тестовые данные)', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)

# Добавляем коэффициент корреляции
correlation = np.corrcoef(comparison_df['actual_price'], comparison_df['predicted_price'])[0, 1]
plt.figtext(0.5, 0.01, f'Корреляция: {correlation:.4f}',
            ha="center", fontsize=12, bbox={"facecolor": "lightblue", "alpha": 0.3, "pad": 5})

plt.tight_layout()
plt.show()

# Выводим статистику предсказаний
print("\nСтатистика предсказаний:")
print(f"Количество тестовых записей: {len(comparison_df)}")
print(f"Средняя фактическая цена: {comparison_df['actual_price'].mean():.4f}")
print(f"Средняя предсказанная цена: {comparison_df['predicted_price'].mean():.4f}")
print(f"Средняя абсолютная ошибка: {mae:.4f}")
print(f"Корреляция: {correlation:.4f}")

# Сохраняем полную информацию о сравнении
comparison_df.to_csv("full_comparison.csv", index=False, sep=',')