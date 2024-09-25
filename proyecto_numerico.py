from tkinter.simpledialog import askfloat, askinteger
from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  NavigationToolbar2Tk)
import numpy as np

def jacobi(A, B, error):
  n = len(B)
  x = np.zeros(n)
  error_norm = float('inf')

  while error_norm > error:
      x_new = np.zeros(n)

      for i in range(n):
          sum_term = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
          x_new[i] = (B[i] - sum_term) / A[i, i]

      error_norm = np.linalg.norm(x_new - x, np.inf)
      x = x_new

  return x

def gauss_elimination(A, b, pivoting='none'):
  # Tamaño del sistema
  n, m = A.shape
  assert n == m, 'La matriz A debe ser cuadrada'
  assert n == len(b), 'El tamaño del vector b debe coincidir con A'

  # Matriz aumentada
  Aug = np.hstack((A, b.reshape(-1, 1)))
  ops = {'adds': 0, 'mults': 0, 'divs': 0}

  # Escalado para pivoteo escalado
  if pivoting == 'scaled':
      s = np.max(np.abs(A), axis=1)
  else:
      s = None

  # Eliminación Gaussiana
  for k in range(n-1):
      if pivoting == 'partial':
          i_max = np.argmax(np.abs(Aug[k:n, k])) + k
          if i_max != k:
              Aug[[k, i_max], :] = Aug[[i_max, k], :]
      elif pivoting == 'scaled':
          i_max = np.argmax(np.abs(Aug[k:n, k] / s[k:n])) + k
          if i_max != k:
              Aug[[k, i_max], :] = Aug[[i_max, k], :]
              s[[k, i_max]] = s[[i_max, k]]

      for i in range(k+1, n):
          factor = Aug[i, k] / Aug[k, k]
          ops['divs'] += 1
          Aug[i, k:n+1] -= factor * Aug[k, k:n+1]
          ops['adds'] += (n - k + 1)
          ops['mults'] += (n - k + 1)

  # Sustitución hacia atrás
  x = np.zeros(n)
  for i in range(n-1, -1, -1):
      x[i] = (Aug[i, -1] - np.dot(Aug[i, i+1:n], x[i+1:n])) / Aug[i, i]
      ops['divs'] += 1
      ops['adds'] += (n - i)
      ops['mults'] += (n - i)

  return x


def sol_estacionaria(L, num_nodos, T_contorno, alpha, op):
  deltax = L/num_nodos
  matriz = np.zeros((num_nodos, num_nodos))
  vec = np.zeros(num_nodos)
  beta = -2-alpha**2*deltax**2

  for i in range(1, num_nodos-1):
    matriz[i,i-1] = 1
    matriz[i,i] = beta
    matriz[i,i+1] = 1

  if op == 1: #Temperatura fija del primer y ultimo nodo
    t_fin = askfloat("Input", "Ingrese la temperatura del ultimo nodo: ")

    f1 = np.zeros(num_nodos)
    f2 = np.zeros(num_nodos)

    f1[0] = 1

    f2[-1] = beta
    f2[-2] = 1


    vec[0] = T_contorno
    vec[-1] = -t_fin

    matriz[0] = f1
    matriz[-1] = f2

    #solucionar sistema matricial
    x = gauss_elimination(matriz, vec, pivoting='none')

    return x

  elif op == 2: # Temperatura fija del primer nodo y derivada igualada a cero del último nodo.
    f1 = np.zeros(num_nodos)
    f2 = np.zeros(num_nodos)

    f1[0] = 1
    f2[-2] = beta
    f2[-3] = 2

    matriz[0] = f1
    matriz[-1] = f2
    vec[0] = T_contorno

    #solucionar sistema matricial
    x = gauss_elimination(matriz, vec, pivoting='none')

    return x

  elif op == 3: # Derivada igualada a cero del primer nodo y temperatura fija del último nodo.
    f1 = np.zeros(num_nodos)
    f2 = np.zeros(num_nodos)

    f1[-1] = 1
    f2[0] = -1
    f2[2] = 1

    matriz[0] = f1
    matriz[-1] = f2
    vec[0] = T_contorno

    #solucionar sistema matricial
    x = gauss_elimination(matriz, vec, pivoting='none')

    return x


def sol_transitoria(L, num_nodos, T_contorno, alpha, tiempo, num_nodos_tiempo, T_inicial, op):
  dx = L / (num_nodos)
  dt = tiempo / (num_nodos_tiempo)
  beta1 = -dt/dx**2
  beta2 = 1+alpha**2*dt+2*dt/dx**2

  A = np.zeros((num_nodos, num_nodos)) # Matriz A
  for i in range(1, num_nodos-1):
    A[i,i-1] = beta1
    A[i,i] = beta2
    A[i,i+1] = beta1

  temps = np.zeros((num_nodos_tiempo, num_nodos))


  if op == 1: ## Temperatura fija del primer y ultimo nodo
    A[0,0] = 1
    A[-1,-1] = 1

    T_contorno2 = askfloat("Input", "Ingrese la temperatura del ultimo nodo: ")

    temps[:, 0] = T_contorno
    temps[:, -1] = T_contorno2
    temps[0, 1:-1] = T_inicial


    for i in range(1, num_nodos_tiempo):
      x = jacobi(A, temps[i - 1, :].T, 0.0001)
      temps[i, :] = x

    return temps

  elif op == 2: ## Temperatura fija del primer nodo y derivada igualada a cero del último nodo.
    A[0,0] = 1

    A[num_nodos-1,num_nodos-1] = beta2
    A[num_nodos-1,num_nodos-2] = 2*beta1

    temps[:, 0] = T_contorno
    temps[0, 1:] = T_inicial


    for i in range(1, num_nodos_tiempo):
      x = jacobi(A, temps[i - 1, :].T, 0.0001)
      temps[i, :] = x

    return temps

  elif op == 3: ## Derivada igualada a cero del primer nodo y temperatura fija del ultimo nodo
    A[0,-1] = 1

    A[-1,0] = beta2
    A[-1,1] = 2*beta1


    temps[:, 0] = T_contorno
    temps[0, 1:] = T_inicial


    for i in range(1, num_nodos_tiempo):
      x = jacobi(A, temps[i - 1, :].T, 0.0001)
      temps[i, :] = x

    return temps

  else:
    exit()






def estacion():
  L = askfloat("Input", "Ingrese la longitud de la superficie de difusión: ")
  num_nodos = askinteger("Input", "Ingrese el número de nodos en el espacio: ")
  alpha = askfloat("Input", "Ingrese la constante del material: ")
  T_contorno = askfloat("Input", "Ingrese la temperatura del contorno: ")
  opcion = askinteger("Input", "Ingrese la opcion\n1) Temperatura fija del primer y ultimo nodo\n2) Temperatura fija del primer nodo y derivada igualada a cero del último nodo\n3) Derivada igualada a cero del primer nodo y temperatura fija del último nodo: ")

  x = sol_estacionaria(L, num_nodos, T_contorno, alpha, opcion)

  #grafica
  fig = Figure(figsize = (75, 75), dpi = 80) 
  plot1 = fig.add_subplot(111)
  plot1.plot(np.linspace(0, L, len(x)), x, '*-')
  plot1.set_xlabel("Posicion")
  plot1.set_ylabel("Temperatura")
  canvas = FigureCanvasTkAgg(fig, master = top)   
  canvas.draw() 
  canvas.get_tk_widget().pack() 

  toolbar = NavigationToolbar2Tk(canvas, top) 
  toolbar.update() 

  canvas.get_tk_widget().pack()



def transito():
  L = askfloat("Input", "Ingrese la longitud de la superficie de difusión: ")
  tiempo = askfloat("Input", "Ingrese el tiempo total de simulación: ")
  num_nodos = askinteger("Input", "Ingrese el número de nodos en el espacio: ")
  num_nodos_tiempo = askinteger("Input", "Ingrese el número de nodos en el tiempo: ")
  alpha = askfloat("Input", "Ingrese la constante del material: ")
  T_contorno = askfloat("Input", "Ingrese la temperatura del contorno: ")
  T_inicial = askfloat("Input", "Ingrese la temperatura inicial (t=0): ")
  opcion = askinteger("Input", "Ingrese la opcion\n1) Temperatura fija del primer y ultimo nodo\n2) Temperatura fija del primer nodo y derivada igualada a cero del último nodo\n3) Derivada igualada a cero del primer nodo y temperatura fija del último nodo: ")

  x = sol_transitoria(L, num_nodos, T_contorno, alpha, tiempo, num_nodos_tiempo, T_inicial, opcion)

  #Grafica
  fig = Figure(figsize = (75, 75), dpi = 80) 
  plot1 = fig.add_subplot(111)
  plot1.plot(np.linspace(0, L, len(x)), x, '*-')
  plot1.set_xlabel("Tiempo")
  plot1.set_ylabel("Temperatura")
  canvas = FigureCanvasTkAgg(fig, master = top)   
  canvas.draw() 
  canvas.get_tk_widget().pack() 

  toolbar = NavigationToolbar2Tk(canvas, top) 
  toolbar.update() 

  canvas.get_tk_widget().pack()


def cerrar():
  top.destroy()


top = Tk()

top.geometry("500x500")

esta = Button(top, text ="Estacionario", command = estacion)
esta.place(x=200,y=50, height=100, width=100)

transi = Button(top, text ="Transitorio", command = transito)
transi.place(x=200,y=175, height=100, width=100)

transi = Button(top, text ="Cerrar", command = cerrar)
transi.place(x=200,y=300, height=100, width=100)

top.mainloop()