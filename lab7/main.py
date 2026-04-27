import numpy as np
import matplotlib.pyplot as plt


# 1. Генерація матриці з діагональним переважанням та вектора f
def generate_system(n=100):
    # Випадкова матриця
    A = np.random.rand(n, n)
    # Забезпечення діагонального переважання [cite: 66]
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1

    # Заданий розв'язок x_i = 2.5 [cite: 67]
    x_true = np.full(n, 2.5)
    # Обчислення b (в методичці названо f) [cite: 67]
    f = A @ x_true

    # Запис у файли
    np.savetxt("matrix_A.txt", A)
    np.savetxt("vector_f.txt", f)
    return A, f


# 2. Функції для зчитування та обчислень [cite: 70]
def load_data():
    A = np.loadtxt("matrix_A.txt")
    f = np.loadtxt("vector_f.txt")
    return A, f


def vector_norm(v):
    return np.max(np.abs(v))


def matrix_norm(C):
    # Обчислення норми матриці за формулою (31) [cite: 31]
    return np.max(np.sum(np.abs(C), axis=1))


# Метод простої ітерації [cite: 22, 27]
def simple_iteration(A, f, eps, x0):
    n = len(f)
    tau = 0.9 / matrix_norm(A)  # Параметр tau в межах (0, 2/||A||)
    C = np.eye(n) - tau * A
    d = tau * f

    x = x0.copy()
    history = []
    for k in range(10000):
        x_new = C @ x + d
        err = vector_norm(x_new - x)
        history.append(err)
        if err < eps:  # Умова закінчення
            return x_new, k + 1, history
        x = x_new
    return x, 10000, history


# Метод Якобі [cite: 38, 44]
def jacobi_method(A, f, eps, x0):
    n = len(f)
    x = x0.copy()
    history = []
    for k in range(10000):
        x_new = np.zeros_like(x)
        for i in range(n):
            # Розгорнута форма методу Якобі [cite: 44]
            s = sum(A[i, j] * x[j] for j in range(n) if i != j)
            x_new[i] = (f[i] - s) / A[i, i]

        err = vector_norm(x_new - x)
        history.append(err)
        if err < eps:
            return x_new, k + 1, history
        x = x_new
    return x, 10000, history


# Метод Зейделя [cite: 49, 58]
def seidel_method(A, f, eps, x0):
    n = len(f)
    x = x0.copy()
    history = []
    for k in range(10000):
        x_old = x.copy()
        for i in range(n):
            # Розгорнута форма методу Зейделя [cite: 58]
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (f[i] - s1 - s2) / A[i, i]

        err = vector_norm(x - x_old)
        history.append(err)
        if err < eps:
            return x, k + 1, history
    return x, 10000, history


# Основний блок виконання [cite: 71, 72, 73]
A, f = generate_system(100)
eps0 = 1e-14
x0 = np.ones(100)  # Початкове наближення x_i = 1.0 [cite: 71]

sol_si, iter_si, hist_si = simple_iteration(A, f, eps0, x0)
sol_ja, iter_ja, hist_ja = jacobi_method(A, f, eps0, x0)
sol_ze, iter_ze, hist_ze = seidel_method(A, f, eps0, x0)

# Вивід результатів
print(f"Метод простої ітерації: {iter_si} ітерацій")
print(f"Метод Якобі:           {iter_ja} ітерацій")
print(f"Метод Зейделя:         {iter_ze} ітерацій")

# Візуалізація збіжності
plt.figure(figsize=(10, 6))
plt.semilogy(hist_si, label='Проста ітерація')
plt.semilogy(hist_ja, label='Якобі')
plt.semilogy(hist_ze, label='Зейделя')
plt.axhline(y=eps0, color='r', linestyle='--', label='Target Precision')
plt.xlabel('Кількість ітерацій')
plt.ylabel('Норма похибки ||X(k+1) - X(k)||')
plt.title('Порівняння швидкості збіжності ітераційних методів')
plt.legend()
plt.grid(True)
plt.show()