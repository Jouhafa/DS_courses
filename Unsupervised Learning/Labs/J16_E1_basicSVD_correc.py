#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#EXERCICE INSPIRED FROM https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#Partie 1 : Decouverte de la  'Singular-value decomposition'
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# define a matrix
A = np.array([[-1, -1], [0, 0.5], [1, 1]])
print(A)

# SVD
U, s, VT = svd(A)
print(U)
print(s)
print(VT)

plt.plot(A[:,0],A[:,1],'x')
plt.show()

#Question 1: Testez la SVD avec
# -> B = np.array([[-1, 1], [0, 0.5], [1, -1]])
# -> C = np.array([[-1, -1], [0, 0.5], [1, -1]])
#Pouvez vous faire un lien entre les éléments U, s, Vt et la manière dont sont
#distribués spatialement les points représentés sur chaque ligne de A ?

B = np.array([[-1, 1], [0, 0.5], [1, -1]])
U, s, VT = svd(B)
print(U)
print(s)
print(VT)
plt.plot(B[:,0],B[:,1],'x')
plt.show()

C = np.array([[-1, -1], [0, 0.5], [1, -1]])
U, s, VT = svd(C)
print(U)
print(s)
print(VT)
plt.plot(C[:,0],C[:,1],'x')
plt.show()

theta=-np.pi/4.  #fits for A (with a 'mirror' effect)
print(np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]))

theta=3*np.pi/4. #fits for B (with a 'mirror' effect)
print(np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]))

theta=2*np.pi/4. #fits for C (with a 'mirror' effect)
print(np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]))

#Question 2 : Une fois la SVD de A effectuee, essayez de reconstruire cette
#             matrice a partir de U, s, VT

# Singular-value decomposition
U, s, VT = svd(A)


[n,p]=A.shape
Sigma = np.zeros((n, p))
Sigma[:p, :p] = np.diag(s)

#Sigma[1,1]=0.

B = U.dot(Sigma.dot(VT))



print('A=',A)
print('B=',B)
