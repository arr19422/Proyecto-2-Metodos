import numpy as np
from FactorizacionMatrices import EliminacionGaussiana
from FactorizacionMatrices import FactorizacionLU
from FactorizacionMatrices import FactorizacionPALU
from FactorizacionMatrices import SolucionLU
import time as tm

def sor(A, b, x, w=1, imax=100, tol=1e-8):
    if len(A.shape) != 2:
        raise ValueError("A must be a 2D array")
    
    n = A.shape[0]
    
    if len(b) != n or len(x) != n:
        raise ValueError("Input dimensions do not match")
    
    flag = False
    k = 0
    
    while not flag and k < imax:
        xk1 = np.zeros(n)
        
        for i in range(n):
            s1 = np.dot(A[i, :i], xk1[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            xk1[i] = (b[i] - s1 - s2) / A[i, i] * w + (1 - w) * x[i]
        
        normal = np.linalg.norm(x - xk1)
        print(f'Iteration : {k+1} -> {normal:.2e}')
        flag = normal < tol
        x = xk1
        k += 1
    
    if k >= imax:
        raise ValueError('The system did not converge')
    
    print("\nSolution:")
    for i, xi in enumerate(x):
        print(f'x{i+1}: {xi:.6f}')

#Con 100 Iteraciones
n=100
dim = 3
A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]])
b = np.array([4, 7, 3])
x0 = np.zeros(len(b))
w = 1.25
imax = 100
tol = 1e-8

#Resuelva 20 sistemas Ax=b utilizando la factorizacion LU

time1 = tm.time()

sor(A, b, x0, w, imax, tol)


time2 = tm.time()

tSOR = time2-time1
print('\nComparación tiempos de Ejecución Solución con SOR, PA-LU y con Eliminación Gaussiana')

print(' ')

print('Tiempo de Resolución ',n, 'sistemas de ecuaciones con SOR :',tSOR, 'segs.')

#Resuelva 20 sistemas Ax=b utilizando PA-LU en cada sistema

[P,L,U] =FactorizacionPALU(A)

time3 = tm.time()

for i in range(0,n):
    b = np.random.rand(dim,1)
    x = SolucionLU(L,U,b)

time4 = tm.time()

tPL = time4- time3

print('\nTiempo de Resolución ',n, 'sistemas con PA-LU:',tPL, 'segs.')

print(' ')

print('PA-LU se tarda en resolver ', tPL/tSOR, 'veces más que el método SOR')


#Resuelva 20 sistemas Ax=b utilizando eliminación Gaussiana en cada sistema

time3 = tm.time()

for i in range(0,n):
    b = np.random.rand(dim,1)
    x = EliminacionGaussiana(A,b)

time4 = tm.time()

tEG = time4- time3

print('\nTiempo de Resolución ',n, 'sistemas con Eliminación Gaussiana:',tEG, 'segs.')

print(' ')

print('La eliminación gaussiana se tarda en resolver ', tEG/tSOR, 'veces más que el método SOR')