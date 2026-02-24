import requests
import numpy as np
import matplotlib.pyplot as plt

# 1-2. Виконання запиту до API (згідно з методичкою)
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

try:
    response = requests.get(url)
    data = response.json()
    results = data["results"]
    if not results:
        raise ValueError("API повернуло порожні дані")
except Exception as e:
    print(f"Помилка зв'язку: {e}")
    exit()

n = len(results)


# 4. Обчислення кумулятивної відстані за формулою гаверсину
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])

distances = [0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)
distances = np.array(distances)

# 3. Запис результатів табуляції у файл
with open("tabulation.txt", "w", encoding="utf-8") as f:
    f.write(f"{'№':<3} | {'Latitude':<10} | {'Longitude':<10} | {'Elev (m)':<8} | {'Dist (m)':<10}\n")
    f.write("-" * 55 + "\n")
    for i in range(n):
        f.write(f"{i:<3} | {coords[i][0]:.6f} | {coords[i][1]:.6f} | {elevations[i]:.2f} | {distances[i]:.2f}\n")


# 6-9. Побудова кубічних сплайнів (Метод прогонки)
def solve_spline(x, y):
    n_pts = len(x)
    h = np.diff(x)

    # 6. Формування трьохдіагональної матриці
    alpha = np.zeros(n_pts)
    beta = np.ones(n_pts)
    gamma = np.zeros(n_pts)
    delta = np.zeros(n_pts)

    for i in range(1, n_pts - 1):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # 7. Пряма та зворотна прогонка
    A = np.zeros(n_pts)
    B = np.zeros(n_pts)
    for i in range(1, n_pts):
        m = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / m
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / m

    c = np.zeros(n_pts)
    c[-1] = B[-1]
    for i in range(n_pts - 2, -1, -1):
        c[i] = A[i] * c[i + 1] + B[i]

    # 8-9. Обчислення коефіцієнтів a, b, d
    a_coeff = y[:-1]
    d_coeff = np.diff(c) / (3 * h)
    b_coeff = (np.diff(y) / h) - (h / 3) * (c[1:] + 2 * c[:-1])

    return a_coeff, b_coeff, c[:-1], d_coeff


# 10. Побудова графіків для різної кількості вузлів
plt.figure(figsize=(12, 7))
node_counts = [10, 15, 20]

for k in node_counts:
    # Використовуємо linspace для вибору індексів, щоб уникнути ділення на 0
    indices = np.linspace(0, n - 1, k, dtype=int)
    x_k = distances[indices]
    y_k = elevations[indices]

    ak, bk, ck, dk = solve_spline(x_k, y_k)
#12.Побудова сплайну
    x_plot = np.linspace(x_k[0], x_k[-1], 300)
    y_plot = []
    for val in x_plot:
        idx = np.searchsorted(x_k, val) - 1
        idx = max(0, min(idx, len(ak) - 1))
        dx = val - x_k[idx]
        y_val = ak[idx] + bk[idx] * dx + ck[idx] * (dx ** 2) + dk[idx] * (dx ** 3)
        y_plot.append(y_val)

    plt.plot(x_plot, y_plot, label=f'{k} вузлів')

# 5. Візуалізація вихідних точок
plt.scatter(distances, elevations, color='red', s=15, label='Вихідні дані API')
plt.title('Інтерполяція профілю висоти маршруту на Говерлу')
plt.xlabel('Відстань від старту (м)')
plt.ylabel('Висота над рівнем моря (м)')
plt.legend()
plt.grid(True)
plt.show()

# --- Побудова сплайна для максимальної кількості вузлів (20) ---
a, b, c_short, d = solve_spline(distances, elevations)
# c_short — це коефіцієнти c для інтервалів, нам також потрібен повний масив c для розрахунків
c = np.append(c_short, 0) # додаємо останній нуль для крайової умови

print(f"\nКоефіцієнти першого сегмента сплайна: a={a[0]:.2f}, b={b[0]:.4f}, c={c[0]:.4f}, d={d[0]:.6f}")

# --- ДОДАТКОВІ ПУНКТИ МЕТОДИЧКИ ---

def get_spline_val(x_nodes, val, a_c, b_c, c_c, d_c):
    idx = np.searchsorted(x_nodes, val) - 1
    idx = max(0, min(idx, len(a_c)-1))
    dx_val = val - x_nodes[idx]
    return a_c[idx] + b_c[idx]*dx_val + c_c[idx]*(dx_val**2) + d_c[idx]*(dx_val**3)

# 1. Загальні характеристики
total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n))

print(f"\n--- Характеристики маршруту ---")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"Сумарний спуск (м): {total_descent:.2f}")

# 2. Аналіз градієнта
xx = np.linspace(distances[0], distances[-1], 1000)
# Тепер передаємо змінні a, b, c, d, які ми визначили вище
yy_full = np.array([get_spline_val(distances, v, a, b, c, d) for v in xx])

dx_step = xx[1] - xx[0]
grad_full = np.gradient(yy_full, dx_step) * 100

print(f"\n--- Аналіз градієнта ---")
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}%")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}%")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}%")

# 3. Механічна енергія
mass = 80
energy = mass * 9.81 * total_ascent
print(f"\n--- Енергія ---")
print(f"Механічна енергія підйому (кДж): {energy/1000:.2f}")