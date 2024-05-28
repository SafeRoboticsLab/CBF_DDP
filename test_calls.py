
import numpy as np


def func(a):
    a[0, 0] = 3
    return np.array(a[1, :])


lista = np.array([[1, 2], [3, 4]])

testa = func(lista)
testa[1] = 5
print(lista)
