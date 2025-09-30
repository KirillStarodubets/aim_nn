import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, gammainc, gamma

def G_l(beta, L, mu):
    """
    Вычисляет функцию G_l с биномиальными коэффициентами и неполной гамма-функцией.
    """
    prefactor = math.sqrt(2 * L + 1) / (1 + math.exp(-beta * mu))
    
    sum_part = 0.0
    
    for k in range(L):
        binom1 = comb(L, k)
        binom2 = comb(L + k, k)
        gamma_part =  gammainc(k + 1, mu * beta)
        term = (-1)**(k+1) * binom1 * binom2 * gamma_part / (beta ** k) / (mu ** k)
        sum_part += term
    
    return prefactor * sum_part

# Параметры
beta = 10
mu = 1.0
a=0
b=13
L_values = np.arange(a, b)

# Вычисление G_l для разных L
results = []
for L in L_values:
    results.append(G_l(beta, L, mu))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(L_values, results, 'o-', markersize=8, linewidth=2)
plt.xlabel('L', fontsize=14)
plt.ylabel('G_l', fontsize=14)
plt.title('Зависимость G_l от l\n( beta=10, mu=1.0)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 10])
plt.show()


# In[]
import math

def calculate_moment_p(p: int, L: int, G_l: list, beta: float) -> float:
    """
    Вычисляет момент c_p по заданной формуле.

    :param p: порядок момента
    :param L: максимальное значение l для суммирования
    :param G_l: список значений G_l, где индекс соответствует l
    :param beta: параметр β
    :return: значение момента c_p
    """
    c_p = 0.0

    for l in range(L + 1):
        # Проверяем условие delta_{p+l, odd}
        if (p + l) % 2 != 1:
            continue

        # Вычисляем факториалы
        numerator = math.factorial(l + p - 1)
        denominator = math.factorial(p - 1) * math.factorial(l - p + 1)

        # Вычисляем t_l^{(p)}
        t_lp = ((-1) ** p) * 2 * math.sqrt(2 * l + 1) * (numerator / denominator)

        # Добавляем вклад в сумму
        c_p += t_lp * G_l[l]

    # Нормируем на β^p
    c_p /= (beta ** p)

    return c_p

def calculate_moments(L: int, G_l: list, beta: float) -> tuple:
    """
    Вычисляет моменты c1, c2, c3 по заданным формулам.

    :param L: максимальное значение l для суммирования
    :param G_l: список значений G_l, где индекс соответствует l
    :param beta: параметр β
    :return: кортеж (c1, c2, c3)
    """
     
    
    c1 = 0.0
    for l in range(0, L + 1, 2):  # только четные l >= 0
        term = (2 *  np.sqrt(2 * l + 1) / beta) * G_l[l]
        c1 -= term

    c2 = 0.0
    for l in range(1, L + 1, 2):  # только нечетные l > 0
        term = (2 *  np.sqrt(2 * l + 1) / (beta ** 2)) * G_l[l] * l * (l + 1)
        c2 += term

    c3 = 0.0
    for l in range(0, L + 1, 2):  # только четные l >= 0
        if l >= 2:  # чтобы избежать отрицательных значений в l(l-1)
            term = ( np.sqrt(2 * l + 1) / (beta ** 3)) * G_l[l] * (l + 2) * (l + 1) * l * (l - 1)
            c3 -= term

    return c1, c2, c3


L = 10
p = 2
beta = 1.0

G_l = [0.1 * (i + 1) for i in range(L + 1)]  # пример значений G_l

cp = calculate_moment_p(p, L, G_l, beta)
print(f"c_{p} = {cp}")

c1, c2, c3 = calculate_moments(L, G_l, beta)
print(f"c1 = {c1}, c2 = {c2}, c3 = {c3}")
