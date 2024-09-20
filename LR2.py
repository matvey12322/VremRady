import pandas as pd
import matplotlib.pyplot as plt

# Подгрузка данных из CSV файла
df = pd.read_csv('passengers2.csv')

# Восполнение предыдущими значениями с помощью метода shift
df['target_filled_forward'] = df['target'].fillna(df['target'].shift(1))

# Восполнение данных с помощью скользящего среднего (ручной код)
def rolling_mean(series, window):
    return series.rolling(window=window, min_periods=1).mean()

# Восполнение пропусков с помощью скользящего среднего
df['target_rolling_mean'] = df['target'].fillna(rolling_mean(df['target'].fillna(method='ffill'), window=3))

# Восполнение пропусков с помощью метода rolling и mean
df['target_rolling_mean_pandas'] = df['target'].fillna(df['target'].rolling(window=10, min_periods=1).mean())

# Линейная интерполяция (ручной код)
def linear_interpolation(series):
    interpolated = series.copy()
    for i in range(len(series)):
        if pd.isna(series[i]):
            # Найти предыдущее и следующее непустое значение
            prev_value = next_value = None
            prev_index = next_index = None
            for j in range(i, -1, -1):
                if not pd.isna(series[j]):
                    prev_value = series[j]
                    prev_index = j
                    break
            for j in range(i, len(series)):
                if not pd.isna(series[j]):
                    next_value = series[j]
                    next_index = j
                    break
            if prev_value is not None and next_value is not None:
                interpolated[i] = prev_value + (next_value - prev_value) * (i - prev_index) / (next_index - prev_index)
    return interpolated

df['target_interpolated_manual'] = linear_interpolation(df['target'])

# Линейная интерполяция pandas
df['target_interpolated'] = df['target'].interpolate(method='linear')

# Простое экспоненциальное сглаживание
def exponential_smoothing(series, alpha):
    smoothed = [series[0]]  # Начальное значение
    for i in range(1, len(series)):
        smoothed.append(alpha * series[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed

alpha = 0.5  # Коэффициент сглаживания
df['reference_smoothed'] = exponential_smoothing(df['reference'], alpha)

plt.figure(figsize=(14, 10))

# График для восполнения предыдущими значениями
plt.subplot(2, 3, 1)
plt.plot(df['Month'], df['reference'], label='Reference')
plt.plot(df['Month'], df['target_filled_forward'], label='Target (Filled Forward)', alpha=0.7)
plt.title('Filled Forward')
plt.legend()

# График для восполнения данных с помощью скользящего среднего (ручной код)
plt.subplot(2, 3, 2)
plt.plot(df['Month'], df['reference'], label='Reference')
plt.plot(df['Month'], df['target_rolling_mean'], label='Target (Rolling Mean)', alpha=0.7)
plt.title('Rolling Mean (Manual)')
plt.legend()

# График для восполнения данных с помощью метода rolling и mean
plt.subplot(2, 3, 3)
plt.plot(df['Month'], df['reference'], label='Reference')
plt.plot(df['Month'], df['target_rolling_mean_pandas'], label='Target (Rolling Mean Pandas)', alpha=0.7)
plt.title('Rolling Mean (Pandas)')
plt.legend()

# График для линейной интерполяции (ручной код)
plt.subplot(2, 3, 4)
plt.plot(df['Month'], df['reference'], label='Reference')
plt.plot(df['Month'], df['target_interpolated_manual'], label='Target (Interpolated Manual)', alpha=0.7)
plt.title('Linear Interpolation (Manual)')
plt.legend()

# График для линейной интерполяции (pandas)
plt.subplot(2, 3, 5)
plt.plot(df['Month'], df['reference'], label='Reference')
plt.plot(df['Month'], df['target_interpolated'], label='Target (Interpolatedl)', alpha=0.7)
plt.title('Linear Interpolation (Pandas)')
plt.legend()


# График для сглаженных данных
plt.subplot(2, 3, 6)
plt.plot(df['Month'], df['reference'], label='Reference')
plt.plot(df['Month'], df['reference_smoothed'], label='Reference (Smoothed)', alpha=0.7)
plt.title('Exponential Smoothing')
plt.legend()

plt.tight_layout()
plt.show()