import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# 1. МАТЕМАТИЧНІ ФУНКЦІЇ (ІНТЕРПОЛЯЦІЯ)

def get_divided_diff_table(x, y):
    """Побудова повної таблиці розділених різниць (Пункт 2 ходу роботи)"""
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    return table


def newton_interpolation(x_nodes, diff_table, x_val):
    """Обчислення значення многочлена Ньютона (Пункт 3 ходу роботи)"""
    n = len(x_nodes)
    coefs = diff_table[0, :]  # Перший рядок таблиці - це коефіцієнти
    res = coefs[0]
    product = 1.0
    for i in range(1, n):
        product *= (x_val - x_nodes[i - 1])
        res += coefs[i] * product
    return res


def lagrange_interpolation(x_nodes, y_nodes, x_val):
    """Метод Лагранжа для порівняння (Дослідницька частина)"""
    n = len(x_nodes)
    res = 0
    for i in range(n):
        basis = 1
        for j in range(n):
            if i != j:
                basis *= (x_val - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        res += y_nodes[i] * basis
    return res



# 2. ОСНОВНА ПРОГРАМА (ВАРІАНТ 1)

def main_task():
    print("=== ЕТАП 1: ОСНОВНЕ ЗАВДАННЯ (ВАРІАНТ 1) ===")

    # Дані Варіанту 1 (можна записати в data.csv або задати списками)
    n_nodes = np.array([1000, 2000, 4000, 8000, 16000])
    t_nodes = np.array([3, 5, 11, 28, 85])

    # 2. Побудова та вивід таблиці
    table = get_divided_diff_table(n_nodes, t_nodes)
    print("\nТаблиця розділених різниць:")
    df_table = pd.DataFrame(table, index=n_nodes, columns=[f"Порядок {i}" for i in range(len(n_nodes))])
    print(df_table.replace(0, ""))  # Заміна нулів для чистоти виводу

    # 3. Прогноз для n=6000
    target_n = 6000
    t_pred_newton = newton_interpolation(n_nodes, table, target_n)
    t_pred_lagrange = lagrange_interpolation(n_nodes, t_nodes, target_n)

    print(f"\nПрогноз для n={target_n}:")
    print(f"Метод Ньютона: {t_pred_newton:.2f} мс")
    print(f"Метод Лагранжа: {t_pred_lagrange:.2f} мс")

    # 4. Графік (Пункт 4 ходу роботи)
    x_range = np.linspace(min(n_nodes), max(n_nodes), 200)
    y_newton = [newton_interpolation(n_nodes, table, xi) for xi in x_range]

    plt.figure(figsize=(10, 5))
    plt.plot(x_range, y_newton, 'b-', label='Поліном Ньютона')
    plt.scatter(n_nodes, t_nodes, color='red', label='Вузли (дані)')
    plt.scatter(target_n, t_pred_newton, color='green', marker='x', s=100, label='Прогноз')
    plt.title("Прогноз часу виконання (Варіант 1)")
    plt.xlabel("Кількість об'єктів (n)")
    plt.ylabel("Час (t, мс)")
    plt.legend()
    plt.grid(True)
    plt.show()



# 3. ДОСЛІДНИЦЬКА ЧАСТИНА

def research_part():
    print("\n" + "=" * 40)
    print("ЕТАП 2: ДОСЛІДНИЦЬКА РОБОТА")
    print("=" * 40)

    # 1. Тестова функція для аналізу (класичний приклад для ефекту Рунге)
    f_test = lambda x: 1 / (1 + x ** 2)
    interval = (-5, 5)

    # Кількість вузлів для дослідження згідно з методичкою [cite: 162, 211-212]
    node_counts = [5, 10, 20]

    # Налаштування графіків
    plt.figure(figsize=(12, 10))
    x_high_res = np.linspace(interval[0], interval[1], 500)
    y_true = f_test(x_high_res)

    for i, n in enumerate(node_counts):
        # Генерація рівномірних вузлів [cite: 76, 81]
        x_nodes = np.linspace(interval[0], interval[1], n)
        y_nodes = f_test(x_nodes)

        # Побудова таблиці розділених різниць та інтерполяція [cite: 156, 159]
        table = get_divided_diff_table(x_nodes, y_nodes)
        y_interp = np.array([newton_interpolation(x_nodes, table, xi) for xi in x_high_res])

        # Обчислення абсолютної похибки [cite: 62, 159]
        errors = np.abs(y_true - y_interp)
        max_err = np.max(errors)

        # --- ВИВІД ТЕКСТУ В ТЕРМІНАЛ ---
        print(f"\nДослідження для n = {n} вузлів:")
        print(f"  - Максимальна абсолютна похибка: {max_err:.4e}")
        if n == 20:
            print("  - Спостереження: помітно різке зростання похибки на краях інтервалу (Ефект Рунге).")

        # --- ПОБУДОВА ГРАФІКІВ ---
        # Графік самої інтерполяції
        plt.subplot(2, 1, 1)
        plt.plot(x_high_res, y_interp, label=f'n={n} (поліном)')
        if i == 0:
            plt.plot(x_high_res, y_true, 'k--', alpha=0.3, label='Оригінальна f(x)')

        # Графік похибок (логарифмічна шкала)
        plt.subplot(2, 1, 2)
        plt.plot(x_high_res, errors, label=f'Похибка n={n}')

    # Оформлення верхнього графіка
    plt.subplot(2, 1, 1)
    plt.title("Інтерполяція Ньютона при різній кількості вузлів")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)

    # Оформлення нижнього графіка
    plt.subplot(2, 1, 2)
    plt.title("Графік похибок (Аналіз ефекту Рунге)")
    plt.yscale('log')  # Логарифмічна шкала для наочності
    plt.xlabel("x")
    plt.ylabel("Абсолютна похибка (log)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.show()


# Запуск всього проекту
if __name__ == "__main__":
    main_task()
    research_part()