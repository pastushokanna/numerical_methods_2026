import numpy as np
import matplotlib.pyplot as plt


# 1. Задана функція з методички
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a, b = 0, 24


# 3. Реалізація складової формули Сімпсона
def simpson_rule(f, a, b, n):
    if n % 2 != 0: n += 1  # n має бути парним
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Формула: (h/3) * (f0 + 4*sum(непарні) + 2*sum(парні) + fn)
    s = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * s


# 2. "Точне" значення (пункт 2 ходу роботи)
# Обчислюємо з дуже великим N для високої точності
I0 = simpson_rule(f, a, b, 100000)
print(f"Прийняте точне значення I0: {I0:.12f}")

# 4. Дослідження залежності точності від N
Ns = np.arange(10, 1001, 10)
errors = []
target_eps = 1e-12
N_opt = None

for n in Ns:
    val = simpson_rule(f, a, b, n)
    err = abs(val - I0)
    errors.append(err)
    if N_opt is None and err <= target_eps:
        N_opt = n

print(f"Оптимальне N_opt: {N_opt}")

# 6. Метод Рунге-Ромберга
# Виберемо N0 кратне 8
N0 = 80
I_h = simpson_rule(f, a, b, N0)
I_2h = simpson_rule(f, a, b, N0 // 2)
# Коефіцієнт 15 для 4-го порядку (2^4 - 1)
I_Runge = I_h + (I_h - I_2h) / 15
print(f"Уточнення Рунге-Ромберга: {I_Runge:.12f}")

# 7. Метод Ейткена
I1 = simpson_rule(f, a, b, N0)
I2 = simpson_rule(f, a, b, N0 * 2)
I3 = simpson_rule(f, a, b, N0 * 4)
I_Aitken = (I2 ** 2 - I1 * I3) / (2 * I2 - (I1 + I3))
p = np.log(abs((I3 - I2) / (I2 - I1))) / np.log(2)
print(f"Метод Ейткена: {I_Aitken:.12f}, Порядок p: {p:.2f}")


# 9. Адаптивний алгоритм
def adaptive_simpson(f, a, b, eps):
    h = b - a
    mid = (a + b) / 2
    I1 = (h / 6) * (f(a) + 4 * f(mid) + f(b))

    def recursive_step(a, b, eps, I_prev):
        m = (a + b) / 2
        h = (b - a)
        # Обчислюємо дві половинки [cite: 131]
        I_left = (h / 12) * (f(a) + 4 * f((a + m) / 2) + f(m))
        I_right = (h / 12) * (f(m) + 4 * f((m + b) / 2) + f(b))
        I2 = I_left + I_right
        if abs(I2 - I_prev) <= 15 * eps:
            return I2 + (I2 - I_prev) / 15

        return recursive_step(a, m, eps / 2, I_left) + recursive_step(m, b, eps / 2, I_right)


    return recursive_step(a, b, eps, I1)

I_adapt = adaptive_simpson(f, a, b, 1e-12)
print(f"Адаптивний метод: {I_adapt:.12f}")

# --- Графік 1: Сама функція навантаження ---
plt.figure(figsize=(10, 5))
x_plot = np.linspace(a, b, 1000)
plt.plot(x_plot, f(x_plot), label='f(x) - навантаження')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час (год)')
plt.ylabel('Навантаження')
plt.grid(True)
plt.legend()

# --- Графік 2: Похибка  ---
plt.figure(figsize=(10, 5))
plt.semilogy(Ns, errors, 'b-o', markersize=4, label='Похибка Сімпсона')
plt.axhline(y=target_eps, color='r', linestyle='--', label='Точність 1e-12')
plt.title('Залежність похибки від числа розбиття N')
plt.xlabel('N')
plt.ylabel('Похибка (log scale)')
plt.grid(True, which="both", ls="-")
plt.legend()

plt.show()
