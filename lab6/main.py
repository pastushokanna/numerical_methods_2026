import numpy as np
import matplotlib.pyplot as plt


# 1. Генерація матриці А та вектора B
def generate_system(n=100, x_val=2.5):
    # Випадкова матриця n x n
    A = np.random.uniform(1, 100, (n, n))
    # Заданий розв'язок x_j = 2.5
    X_true = np.full(n, x_val)
    # Обчислення вектора вільних членів
    B = A @ X_true

    # Запис у текстові файли
    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_B.txt', B)
    return n


# 2. Функції для роботи з LU-розкладом
def read_data():
    A = np.loadtxt('matrix_A.txt')
    B = np.loadtxt('vector_B.txt')
    return A, B


def lu_decomposition(A, n):
    # Створення матриць L та U
    L = np.zeros((n, n))
    U = np.eye(n)  # Діагональні елементи U = 1

    for k in range(n):
        # Обчислення k-го стовпця L
        for i in range(k, n):
            sum_lu = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_lu

        # Обчислення k-го рядка U
        for i in range(k + 1, n):
            sum_lu = sum(L[k][j] * U[j][i] for j in range(k))
            U[k][i] = (A[k][i] - sum_lu) / L[k][k]
    return L, U


def solve_lu(L, U, B, n):
    # Розв'язок LZ = B (прямий хід)
    Z = np.zeros(n)
    for k in range(n):
        sum_lz = sum(L[k][j] * Z[j] for j in range(k))
        Z[k] = (B[k] - sum_lz) / L[k][k]

    # Розв'язок UX = Z (зворотний хід)
    X = np.zeros(n)
    X[n - 1] = Z[n - 1]
    for k in range(n - 2, -1, -1):
        sum_ux = sum(U[k][j] * X[j] for j in range(k + 1, n))
        X[k] = Z[k] - sum_ux
    return X


def get_norm(vector):
    # Обчислення норми вектора (max модуль)
    return np.max(np.abs(vector))


def main():
    N = 100
    eps_0 = 1e-14  # Цільова точність

    # Крок 1: Підготовка даних
    generate_system(N)
    A, B = read_data()

    # Крок 2: LU-розклад та запис у файл
    L, U = lu_decomposition(A, N)
    np.savetxt('lu_decomposition.txt', np.hstack((L, U)))

    # Крок 3: Початковий розв'язок
    X_current = solve_lu(L, U, B, N)

    # Крок 4: Оцінка початкової точності
    initial_residual = A @ X_current - B
    print(f"Початкова похибка (max|AX-B|): {get_norm(initial_residual):.2e}")

    # Крок 5: Ітераційне уточнення
    residuals = []
    iterations = 0
    max_iter = 20

    print("\nПочаток ітераційного уточнення:")
    while iterations < max_iter:
        # Обчислення вектора нев'язки R = B - AX
        R = B - A @ X_current
        norm_r = get_norm(R)
        residuals.append(norm_r)

        print(f"Ітерація {iterations + 1}: похибка = {norm_r:.2e}")

        # Перевірка умови зупинки
        if norm_r <= eps_0:
            break

        # Розв'язок допоміжної системи для уточнення
        delta_X = solve_lu(L, U, R, N)
        X_current = X_current + delta_X
        iterations += 1

    print(f"\nФінальна кількість ітерацій: {len(residuals)}")
    print(f"Середнє значення знайденого X: {np.mean(X_current):.4f}")

    # Побудова графіка збіжності
    plt.figure(figsize=(9, 5))
    plt.semilogy(range(1, len(residuals) + 1), residuals, 'bo-', linewidth=2, markersize=6)
    plt.axhline(y=eps_0, color='r', linestyle='--', label=f'Межа точності ({eps_0})')
    plt.title("Збіжність методу")
    plt.xlabel("Номер ітерації")
    plt.ylabel("Норма нев'язки (логарифмічна шкала)")
    plt.show()


if __name__ == "__main__":
    main()