import numpy as np
import time as tm

def EliminacionGaussiana(A,b,pasos='N'):

    A=np.asfarray(A)   #Si las entradas son enteras, se considera una matriz con decimales
    #Resuelve el sistema de ecuaciones utilizando eliminación gaussiana sin pivoteo

    # Dimension de la matriz A
    m = np.shape(A)[0]  # número de filas
    n = np.shape(A)[1]  # número de columnas

    if m != n:
        print('La matriz debe ser cuadrada para realizar la eliminación gaussiana')

    Aug = np.append(A, b, 1)  # Construya la matriz aumentada de A

    # Eliminación hacia adelante
    for k in range(0, n-1):
        for i in range(k + 1, n):
            factor = Aug[i][k] / Aug[k][k]  # Encuentre el multiplicador
            # Elimine las entradas debajo de la k-ésima fila de A
            #for j in range(k,n+1):
                #Aug[i][j]= Aug[i][j] - factor * Aug[k][j]   #Actualice la matriz aumentada
            Aug[i,k:n+1] = Aug[i,k:n+1] - factor * Aug[k,k:n+1]
        if pasos == 'S':
            print('Eliminación hacia adelante de la columna ', k + 1, '\n')
            np.set_printoptions(suppress=True)
            print(Aug)
            print(' ')
            #tm.sleep(3)  # Despliegue los resultados intermedios por tres segundos

    # Sustitución hacia atrás
    x = np.zeros((n, 1))
    x[n - 1] = Aug[n - 1][n] / Aug[n - 1][n - 1]  # El vector del lado derecho es Aug(:, nb)

    for i in range(n - 2, -1, -1):
        x[i] = (Aug[i][n] - np.dot(Aug[i][i+1:n], x[i + 1:n])) / Aug[i][i]
    # Fin de la sustitución hacia atrás
    return x


#######################################
#######################################


def EliminacionGaussPivoteo(A,b,pasos='N'):

    #Resuelve el sistema de ecuaciones utilizando eliminación gaussiana sin pivoteo

    A = np.asfarray(A)  # Si las entradas son enteras, se considera una matriz con decimales

    # Dimension de la matriz A
    m = np.shape(A)[0]  # número de filas
    n = np.shape(A)[1]  # número de columnas

    if m != n:
        print('La matriz debe ser cuadrada para realizar la eliminación gaussiana')

    Aug = np.append(A, b, 1)  # Construya la matriz aumentada de A

    # Eliminación hacia adelante
    for k in range(0, n-1):
        #Paso de Pivoteo Parcial
        indice= np.argmax(np.abs(Aug[k:n,k]))  #Encuentre la localización de la entrada con mayor longitud
        Aug[[k, indice+k]]= Aug[[indice+k, k]] #Intercambie las filas de la matriz aumentada
        #print('Matriz Aumentada intercambiada')
        #print(Aug)
        for i in range(k + 1, n):
            factor = Aug[i][k] / Aug[k][k]  # Encuentre el multiplicador
            # Elimine las entradas debajo de la k-ésima fila de A
            for j in range(k,n+1):
                Aug[i][j]= Aug[i][j] - factor * Aug[k][j]   #Actualice la matriz aumentada

            #Aug[i][k:n+1] = Aug[i][k:n+1] - factor * Aug[k][k:n+1]

        if pasos == 'S':
            print('Eliminación hacia adelante de la columna ', k + 1, '\n')
            np.set_printoptions(suppress=True)
            print(Aug)
            print(' ')
            tm.sleep(3)  # Despliegue los resultados intermedios por tres segundos

    # Sustitución hacia atrás
    x = np.zeros((n, 1))
    x[n - 1] = Aug[n - 1][n] / Aug[n - 1][n - 1]  # El vector del lado derecho es Aug(:, nb)
    for i in range(n - 2, -1, -1):
        x[i] = (Aug[i][n] - np.dot(Aug[i][i+1:n], x[i + 1:n])) / Aug[i][i]
    # Fin de la sustitución hacia atrás


    return x


#######################################
#######################################

def FactorizacionLU(A,pasos='N'):

    #Encuentra la factorización LU de una matriz

    A = np.asfarray(A)  # Si las entradas son enteras, se considera una matriz con decimales

    # Dimension de la matriz A
    m = np.shape(A)[0]   #número de filas
    n = np.shape(A)[1]   #número de columnas

    #Valores iniciales de las matrices L y U
    L = np.eye(m, n) #Valor inicial de L es la matriz identidad
    U = A #Valor inicial de U es la matriz A


    if m != n:
        print('La matriz debe ser cuadrada para encontrar la factorización LU')

    # Eliminación hacia adelante
    for k in range(0, n-1):
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]  # Encuentre el multiplicador
            L[i][k] = factor            #Guarde el multiplicador en la entrada de L respectiva
            # Elimine las entradas debajo de la k-ésima fila de A
            for j in range(k,n):
                U[i][j]= U[i][j] - factor * U[k][j]   #Actualice la matriz U
            #U[i,k:n] = U[i,k:n] - factor * U[k,k:n]   #Actualice la matriz U
            #U[i][k:n] = U[i][k:n] - factor * U[k][k:n]   #Actualice la matriz U

        if pasos == 'S':
            print('Eliminación hacia adelante de la columna ', k + 1, '\n')
            #np.set_printoptions(precision=4) #Sólo imprime 4 decimales de cada entrada
            np.set_printoptions(suppress=True)
            print(U)
            print(' ')
            tm.sleep(3)  # Despliegue los resultados intermedios por tres segundos

    #Guarde los valores de las matrices L y U
    return [L,U]

#######################################
#######################################

def FactorizacionPALU(A,pasos='N'):

    #Encuentra la factorización LU de una matriz

    A = np.asfarray(A)  # Si las entradas son enteras, se considera una matriz con decimales

    # Dimension de la matriz A
    m = np.shape(A)[0]   #número de filas
    n = np.shape(A)[1]   #número de columnas

    #Valores iniciales de las matrices L y U
    L = np.zeros((m, n)) #Valor inicial de L es la matriz identidad
    U = A #Valor inicial de U es la matriz A
    P = np.eye(m,n)  #Matriz inicial de permutación es la identidad


    if m != n:
        print('La matriz debe ser cuadrada para encontrar la factorización LU')

    # Eliminación hacia adelante
    for k in range(0, n-1):
        # Paso de Pivoteo Parcial
        indice = np.argmax(np.abs(U[k:n, k]))      # Encuentre la localización de la entrada con mayor longitud
        U[[k, indice+k]] = U[[indice+k, k]]        # Intercambie las filas de la matriz U
        L[[k, indice+k]] = L[[indice+k, k]]        # Intercambie las filas de la matriz L
        P[[k, indice+k]] = P[[indice+k, k]]        # Intercambie las filas de la matriz P
        L[k][k] = 1  # Agregue 1 en la diagonal principal de L después de intercambiar filas
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]  # Encuentre el multiplicador
            L[i][k] = factor            #Guarde el multiplicador en la entrada de L respectiva
            # Elimine las entradas debajo de la k-ésima fila de A
            for j in range(k,n):
                U[i][j]= U[i][j] - factor * U[k][j]   #Actualice la matriz U
            #U[i,k:n] = U[i,k:n] - factor * U[k,k:n]   #Actualice la matriz U



        if pasos == 'S':
            print('Eliminación hacia adelante de la columna ', k + 1, '\n')
            #np.set_printoptions(precision=4) #Sólo imprime 4 decimales de cada entrada
            np.set_printoptions(suppress=True)
            print('U')
            print(U)
            print('L')
            print(L)
            print('P')
            print(P)
            print(' ')
            tm.sleep(3)  # Despliegue los resultados intermedios por tres segundos

    L[n-1][n-1]=1 #Agregue un 1 a la última entrada de la diagonal principal
    #Guarde los valores de las matrices L y U
    return [P,L,U]

#######################################
#######################################

def SustitucionAdelante(L,b):
    b=np.asfarray(b)  # Si las entradas son enteras, se considera un vector con decimales
    y=np.zeros_like(b)
    for i in range(len(b)):
        y[i]=(b[i]-np.dot(L[i,0:i],y[0:i]))/L[i,i]

    return y

def SustitucionAtras(U,y):
    y = np.asfarray(y)  # Si las entradas son enteras, se considera un vector con decimales
    x=np.zeros_like(y)
    for i in range(len(x),0,-1):
      x[i-1]=(y[i-1]-np.dot(U[i-1,i:],x[i:]))/U[i-1,i-1]

    return x

def SolucionLU(L,U,b):
    y=SustitucionAdelante(L,b)
    x=SustitucionAtras(U,y)

    return x
