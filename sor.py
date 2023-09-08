import numpy as np

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

# Example usage:
A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]])
b = np.array([4, 7, 3])
x0 = np.zeros(len(b))
w = 1.25
imax = 100
tol = 1e-8

sor(A, b, x0, w, imax, tol)