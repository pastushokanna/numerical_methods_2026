import numpy as np
import matplotlib.pyplot as plt

# 1. Визначення функції та її аналітичної похідної
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def M_prime_exact(t):
    # Аналітична похідна
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# Центральна різницева схема для першої похідної
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Параметри задачі
t0 = 1.0
exact_val = M_prime_exact(t0)

print(f"--- Аналітичне значення ---")
print(f"M'({t0}) точне = {exact_val:.10f}\n")

# 2. Дослідження залежності похибки від кроку h
h_values = np.logspace(-20, 3, num=100)
errors = []

best_h = h_values[0]
min_error = float('inf')

for h in h_values:
    approx = central_diff(M, t0, h)
    error = abs(approx - exact_val)
    errors.append(error)
    if error < min_error:
        min_error = error
        best_h = h

print(f"--- Оптимізація кроку ---")
print(f"Найкращий крок h0: {best_h:.2e}")
print(f"Мінімальна похибка R0: {min_error:.2e}\n")

# Візуалізація залежності похибки (необов'язково, але корисно для звіту)
plt.figure(figsize=(8, 5))
plt.loglog(h_values, errors)
plt.axvline(best_h, color='r', linestyle='--', label=f'h_opt ≈ {best_h:.1e}')
plt.xlabel('Крок h')
# Похибка чисельного диференціювання визначається виразом R = ψh^p + Δ/h
plt.ylabel('Похибка R')
plt.title('Залежність похибки від кроку h (Log-Log scale)')
plt.legend()
plt.grid(True)
plt.show()

# 3-6. Метод Рунге-Ромберга
h_base = 1e-3
d_h = central_diff(M, t0, h_base)      # y'(h)
d_2h = central_diff(M, t0, 2 * h_base)  # y'(2h)

# Формула: y_R = y'(h) + (y'(h) - y'(2h)) / (q^p - 1)
# Оскільки схема центральна, порядок p = 2. Коефіцієнт q = 2. (2^2 - 1) = 3
y_runge = d_h + (d_h - d_2h) / 3
err_runge = abs(y_runge - exact_val)

print(f"--- Метод Рунге-Ромберга (h={h_base}) ---")
print(f"y'(h):   {d_h:.10f} | Похибка R1: {abs(d_h - exact_val):.2e}")
print(f"y'(2h):  {d_2h:.10f}")
print(f"y'_RR:   {y_runge:.10f} | Похибка R2: {err_runge:.2e}")
print(f"Покращення точності: {abs(d_h - exact_val)/err_runge:.2f} разів\n")

# 7. Метод Ейткена
d_4h = central_diff(M, t0, 4 * h_base)

# Уточнене значення за Ейткеном
denom_aitken = 2 * d_2h - (d_4h + d_h)
if denom_aitken != 0:
    y_aitken = (d_2h**2 - d_4h * d_h) / denom_aitken
    # Оцінка порядку точності p
    p_aitken = np.log(abs((d_4h - d_2h) / (d_2h - d_h))) / np.log(2)
else:
    y_aitken, p_aitken = 0, 0

err_aitken = abs(y_aitken - exact_val)

print(f"--- Метод Ейткена ---")
print(f"y'(4h):  {d_4h:.10f}")
print(f"y'_E:    {y_aitken:.10f} | Похибка R3: {err_aitken:.2e}")
print(f"Оцінений порядок точності p: {p_aitken:.2f}")
# Графік швидкості зміни вологості (похідної)
t_plot = np.linspace(0, 20, 400)
y_exact_plot = M_prime_exact(t_plot)
y_approx_plot = [central_diff(M, t, h_base) for t in t_plot]

plt.figure(figsize=(10, 6))
plt.plot(t_plot, y_exact_plot, label="Точна похідна $M'(t)$", color='blue', linewidth=2)
plt.plot(t_plot, y_approx_plot, '--', label=f"Чисельна (h={h_base})", color='red')
plt.axhline(0, color='black', lw=1)
plt.title("Швидкість зміни вологості ґрунту ( Drying Rate )")
plt.xlabel("Час t (дні/години)")
plt.ylabel("M'(t)")
plt.legend()
plt.grid(True)
plt.show()

# Побудова комбінованого графіка для аналізу поливу
fig, ax1 = plt.subplots(figsize=(10, 6))

# Перша вісь (Вологість)
color = 'tab:blue'
ax1.set_xlabel('Час t (год/дні)')
ax1.set_ylabel('Вологість M(t)', color=color)
ax1.plot(t_plot, M(t_plot), color=color, linewidth=2, label='Вологість')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Друга вісь (Швидкість зміни)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("Швидкість зміни M'(t)", color=color)
ax2.plot(t_plot, y_exact_plot, color=color, linestyle='--', label='Швидкість висихання')
ax2.tick_params(axis='y', labelcolor=color)

# Додаємо горизонтальну лінію "критичної швидкості"
ax2.axhline(-4, color='gray', linestyle=':', label='Поріг ввімкнення')

plt.title("Комплексний аналіз стану ґрунту")
fig.tight_layout()
plt.show()
