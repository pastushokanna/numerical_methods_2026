import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Зчитування даних
def load_data():
    # Дані взяті безпосередньо з вашої таблиці в методичці
    data = {
        'Month': list(range(1, 25)),
        'Temp': [-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0, -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3]
    }
    return np.array(data['Month']), np.array(data['Temp'])


# 2. Формування матриць B та С
def form_system(x, f, m):
    n = len(x)
    rho = np.ones(n)  # Ваги за замовчуванням

    # Формування матриці b_kl = сума(rho * x^(k+l))
    B = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for l in range(m + 1):
            B[k, l] = np.sum(rho * (x ** (k + l)))

    # Формування вектора c_k = сума(rho * f * x^k)
    C = np.zeros(m + 1)
    for k in range(m + 1):
        C[k] = np.sum(rho * f * (x ** k))

    return B, C


# 3. Метод Гаусса з вибором головного елемента по стовпцях
def gauss_solve(A, b):
    n = len(b)
    A = A.copy().astype(float)
    b = b.copy().astype(float)

    # Прямий хід з вибором головного елемента
    for k in range(n):
        # Пошук найбільшого по модулю елемента в стовпці
        max_idx = np.argmax(np.abs(A[k:, k])) + k
        # Перестановка рядків
        A[[k, max_idx]] = A[[max_idx, k]]
        b[[k, max_idx]] = b[[max_idx, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотній хід
    x_res = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_res[i] = (b[i] - np.dot(A[i, i + 1:], x_res[i + 1:])) / A[i, i]
    return x_res


# 4. Функція обчислення многочлена
def get_polynomial(x, coef):
    y = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y += c * (x ** i)
    return y


# 5. Функція обчислення дисперсії
def get_variance(y_true, y_approx):
    n = len(y_true)
    # Формула дисперсії з методички
    return np.sqrt(np.sum((y_approx - y_true) ** 2) / n)


# --- ОСНОВНИЙ ХІД РОБОТИ ---

# Завантаження даних
x_data, y_data = load_data()
variances = []
degrees = list(range(1, 11))  # Випадки m=1...10

print("--- Розрахунок дисперсії для різних степенів ---")
for m in degrees:
    B, C = form_system(x_data, y_data, m)
    coef = gauss_solve(B, C)
    y_approx = get_polynomial(x_data, coef)
    var = get_variance(y_data, y_approx)
    variances.append(var)
    print(f"Степінь m={m:2}: Дисперсія = {var:.4f}")

# Вибір оптимального ступеня
opt_m = degrees[np.argmin(variances)]
print(f"\nОптимальний ступінь за мінімумом дисперсії: m={opt_m}")

# Фінальний розрахунок для оптимального ступеня
B_opt, C_opt = form_system(x_data, y_data, opt_m)
coef_opt = gauss_solve(B_opt, C_opt)
y_final = get_polynomial(x_data, coef_opt)

# Прогноз на наступні 3 місяця
x_future = np.array([25, 26, 27])
y_future = get_polynomial(x_future, coef_opt)
print(f"Прогноз на 25-27 місяці: {y_future}")

# --- ПОБУДОВА ГРАФІКІВ  ---
plt.figure(figsize=(10, 8))

# Графік апроксимації
plt.subplot(2, 1, 1)
plt.scatter(x_data, y_data, color='red', label='Фактичні дані (CSV)')
plt.plot(x_data, y_final, color='blue', linewidth=2, label=f'МНК Поліном (m={opt_m})')
plt.scatter(x_future, y_future, color='green', marker='D', label='Прогноз (Екстраполяція)')
plt.title('Найкраще квадратичне наближення')
plt.xlabel('Місяць')
plt.ylabel('Температура')
plt.legend()
plt.grid(True)

# Графік похибки
plt.subplot(2, 1, 2)
# Обчислення похибки epsilon(x) = |f(x) - phi(x)|
error = np.abs(y_data - y_final)
plt.plot(x_data, error, 'o-', color='orange', label='Абсолютна похибка')
plt.title('Графік похибки апроксимації за місяцями')
plt.xlabel('Місяць')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Додатковий графік дисперсії
plt.figure()
plt.plot(degrees, variances, 'ro-')
plt.title('Залежність дисперсії від ступеня полінома')
plt.xlabel('Ступінь m')
plt.ylabel('Дисперсія delta')
plt.grid(True)
plt.show()


# Побудова графіків похибок для всіх m від 1 до 10
plt.figure(figsize=(12, 7))

# крок для похибки h1 = (xn-x0)/(20n)
x_fine = np.linspace(x_data[0], x_data[-1], 20 * len(x_data))

for m in degrees:
    # Розраховуємо коефіцієнти для кожного m
    B_m, C_m = form_system(x_data, y_data, m)
    coef_m = gauss_solve(B_m, C_m)

    # Обчислюємо значення полінома та фактичні значення для порівняння
    y_approx_m = get_polynomial(x_data, coef_m)
    error_m = np.abs(y_data - y_approx_m)

    # Малюємо лінію похибки для кожного ступеня
    plt.plot(x_data, error_m, label=f'm={m}', alpha=0.8)

plt.title('Графіки похибки апроксимації epsilon(x) для m=1...10 (п. 4)', fontsize=12)
plt.xlabel('Місяць (x)', fontsize=10)
plt.ylabel('Абсолютна похибка |f(x) - phi(x)|', fontsize=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Степінь m")
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()