import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc


#EDA
# Функция для загрузки данных и проведения EDA
def load_and_analyze(train_file, test_file, dataset_name):
    # Загрузка данных
    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)

    # Объединение данных для анализа
    combined_df = pd.concat([train_df, test_df], axis=0)

    # Основная информация о данных
    print(f"Информация о данных {dataset_name}:")
    print(combined_df.info())
    print("\nОписание данных:")
    print(combined_df.describe())

    # Проверка наличия пропущенных значений
    print("\nПропущенные значения:")
    print(combined_df.isnull().sum())

    # Визуализация временного ряда с помощью графика скользящего среднего
    combined_df['rolling_mean'] = combined_df[0].rolling(window=10).mean()
    plt.figure(figsize=(14, 7))
    plt.plot(combined_df.index, combined_df[0], label='Оригинальный временной ряд')
    plt.plot(combined_df.index, combined_df['rolling_mean'], label='Скользящее среднее', color='red')
    plt.title(f'Временной ряд и скользящее среднее для {dataset_name}')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.legend()
    plt.show()

    # Визуализация распределения значений
    plt.figure(figsize=(10, 6))
    sns.histplot(combined_df[0], bins=30, kde=True)
    plt.title(f'Распределение значений для {dataset_name}')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.show()

    # Визуализация автокорреляции
    plt.figure(figsize=(10, 6))
    pd.plotting.autocorrelation_plot(combined_df[0])
    plt.title(f'Автокорреляция для {dataset_name}')
    plt.show()

    # Визуализация частичной автокорреляции
    plt.figure(figsize=(10, 6))
    plot_pacf(combined_df[0], lags=50)
    plt.title(f'Частичная автокорреляция для {dataset_name}')
    plt.show()

# Проведение EDA для данных о вине
load_and_analyze('Wine_TRAIN.csv', 'Wine_TEST.csv', 'Wine')

# Проведение EDA для данных о йоге
load_and_analyze('yoga_TRAIN.csv', 'yoga_TEST.csv', 'Yoga')

#ML
# Функция для загрузки данных и построения модели
def load_and_classify(train_file, test_file, dataset_name):
    # Загрузка данных
    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)

    # Предположим, что первый столбец является целевой переменной (меткой класса)
    X_train = train_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]
    y_test = test_df.iloc[:, 0]

    # Инициализация модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Перекрёстная проверка
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Перекрёстная проверка для {dataset_name}:")
    print(f"Средняя точность: {np.mean(cv_scores):.4f}")
    print(f"Стандартное отклонение: {np.std(cv_scores):.4f}")

    # Обучение модели на всех тренировочных данных
    model.fit(X_train, y_train)

    # Предсказание на тестовых данных
    y_pred = model.predict(X_test)

    # Оценка модели
    print(f"Оценка модели для {dataset_name}:")
    print(f"Точность: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Визуализация важности признаков
    feature_importances = model.feature_importances_
    feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'Feature {i}' for i in range(X_train.shape[1])]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_names)
    plt.title(f'Важность признаков для {dataset_name}')
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.show()

    # Визуализация матрицы ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Матрица ошибок для {dataset_name}')
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.show()


# Проведение классификации для данных о вине
load_and_classify('Wine_TRAIN.csv', 'Wine_TEST.csv', 'Wine')

# Проведение классификации для данных о йоге
load_and_classify('yoga_TRAIN.csv', 'yoga_TEST.csv', 'Yoga')
