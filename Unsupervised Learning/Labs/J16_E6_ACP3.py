

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Aspects mathematiques de l'ACP. Comme montr� dans le cours, l'ACP correspond a une d�composition en valeurs principales de la matrice de correlation issue de la matrice observee. Nous allons le v�rifier ici
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

import numpy as np
import matplotlib.pyplot as plt


#1) Nous allons travailler avec une matrice de dimension 5x3. Elle corresponds a n=5 observations en dimension p=3

M=np.array([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[1,0,1]])+np.random.randn(5,3)/5

print(M)



#2) Analyse des composantes principales et de la variance expliquee avec les outils scikit-learn

"""
Question 1 : 
-> Centrer-reduire M
-> effectuer son ACP
-> Visualiser les composantes principales et de la variance expliquee
"""



#3) Comme nous l'avons vu dans le cours, l'ACP est une diagonalisation de la matrice de covariance. On va v�rifier ici que les composantes et la variance expliquee issues de l'ACP de M sont similaires aux informations renvoyees trouvees en calculant la decomposition en valeur principales de la matrice de covariance de M.

"""
Question 2 : resolvez le probleme en calculant
-> La matrice de covariance des observations dans M
-> Utilisez np.linalg.eig pour decomposer la matrice de covariance
-> Mettez en lien les resultats avec pca.explained_variance_ et pca.components_

Remarque : il y a un l�ger bug dans la pca de sklearn. La matrice de covariance qu'elle considere est normalisee avec 'n' et non 'n-1'
"""
