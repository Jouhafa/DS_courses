
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Manipulation du maximum de vraisemblance 2 :
#
# On veut quantifier l'efficacite d'un produit pour ameliorer le rendement d'un moteur.
#
# Les resultats quantifies sont dans le fichier 'J18_E4_QuantifiedData.csv' qui contient:
#   (1ere colonne) La quantite de produit injecte
#   (2eme colonne) Le rendement mesure
#
# Nous allons essayer de trouver la relation entre la quantite de produit injecte et le
# rendement mesure a l'aide d'un modele de regression lineaire avec differentes
# modelisations du bruit
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import matplotlib.pyplot as plt


MyData=np.genfromtxt('J18_E4_QuantifiedData.csv')

plt.scatter(MyData[:,0],MyData[:,1])
plt.show()


#QUESTION 1 : Essayez de resoudre le probleme a l'aide de l'algorithme de
#             regression lineaire de scikit-learn

X=MyData[:,0]
Y=MyData[:,1]

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X[:, np.newaxis], Y)

fig = plt.figure()
plt.plot(X, Y, 'r.')
plt.plot(X, lr.predict(X[:, np.newaxis]), 'b-')
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.title('Linear regression')
plt.show()


# On peut constater que la pente de la courbe est legerement trop
# forte. Cette mauvaise estimation est due a trois observations a
# droite de la figure qui font un effet levier. La regression
# lineaire minimise l'erreur d'approximation au carre sur les
# observations d'apprentissage. De maniere sous-jacente cela se
# base sur l'hypothese que les erreurs d'approximation suivent
# une loi normale centree (et pas forcement reduite). Hors, les
# erreurs d'approximation autour d'un modele lineaire sont
# clairement non symetriques ici. Nous allons alors resoudre le
# probleme au sens du maximum de vraisemblance.

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#QUESTION 2 :
#
#On va modeliser le probleme sous la forme :
#
#-> $ypred_i = a * x_i +b \,,\, \forall i = 1, \ldots, n$
#-> $err_i= ypred_i-y_i$
#
#ou les $x_i$ et $y_i$ sont les donnees d'apprentissage pour les
#observations $i$ dans $[1, 2, ..., n]$, et $ypred_i$ approche
#$y_i$. Les deux parametres du modele lineaire que l'on cherche
#a estimer sont $a$ et $b$. Afin de résoudre le probleme, on va
#alors repondre aux sous-questions suivantes :
#
#Question 2.1 : Codez une fonction qui calcul les erreurs d'approximations
#               pour toutes les observations de $X$ et $Y$ avec un $a$ et
#               un $b$ specifiques.
#Question 2.2 : Codez une fonction qui calcule la vraisemblance de
#               parametres pour lesquel l'erreur d'approximation suit une
#               loi normale centree d'ecart type sigma. On donnera la
#               valeur par defaut sigma=2
#Question 2.3 : Codez une fonction qui calcule la vraisemblance de
#               parametres pour lesquel l'erreur d'approximation suit une loi
#               de chi2. On fixera par defaut le nombre de degres de liberte
#               ddl=3 et l'echelle de la loi (scale) a 0.4. On fera très
#               attention au fait que la densite de probabilite d'une valeur
#               negative sera egale a zero avec la loi du chi2.
#Question 2.4 : Utilisez les fonctions de calcul de la vraisemblance pour
#               trouver une relation lineaire qui semble raisonable, i.e. pour
#               trouver les parametres a et b les plus vraisemblables.
#               On pourra eventuellement s'aider d'une representation du nuage
#               de points qui represente le 'score' attribue a chaque
#               observation.
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#CODE 2.1
def compute_errors(X,Y,theta):
    """
    returns a vector having the same size as X or Y which represents the errors
    with a 1D linear model of parameters theta=[a,b]
    """
    a=theta[0]
    b=theta[1]

    errors=Y-(a*X+b)
    return errors

#CODE 2.2

from scipy.stats import norm

def likelihood_normal(X,Y,theta,sigma=2.,verbose=False):
    """
    returns the likelihood of the 1D linear model with parameters theta=[a,b] and
    the errors following a normal law of std=sigma
    """

    errors=compute_errors(X,Y,theta)

    scores=[]
    for locError in errors:
        scores.append(norm.pdf(locError, loc=0, scale=sigma))

    likelihood=1.
    for score in scores:
        likelihood*=score

    if verbose:
        plt.scatter(X,Y,c=scores,cmap='rainbow')
        plt.plot(X, theta[0]*X+theta[1], 'b-')
        plt.colorbar()
        plt.title('likelihood='+str(likelihood))
        plt.show()

    return likelihood

#CODE 2.3

from scipy.stats import chi2

def likelihood_chi2(X,Y,theta,dof=3,sc=0.4,verbose=False):
    """
    returns the likelihood of the 1D linear model with parameters theta=[a,b] and
    the errors following a chi2 law of dof degrees of freedom
    """

    errors=compute_errors(X,Y,theta)

    scores=[]
    for locError in errors:
        scores.append(chi2.pdf(locError, dof,scale=sc))

    likelihood=1.
    for score in scores:
        likelihood*=score

    if verbose:
        plt.scatter(X,Y,c=scores,cmap='rainbow')
        plt.plot(X, theta[0]*X+theta[1], 'b-')
        plt.colorbar()
        plt.title('likelihood='+str(likelihood))
        plt.show()

    return likelihood

#CODE 2.4

likelihood_normal(X,Y,[1.2,-0.3],sigma=2.,verbose=True)

likelihood_chi2(X,Y,[1.2,-0.6],dof=3,sc=0.4,verbose=True)



#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#QUESTION 3 : Codez une fonction de descente de gradient pour
#             apprendre les parametres optimaux du modele (a et b)
#             avec les deux types de bruit consideres mais leurs
#             parametres fixes aux valeurs par defaut.
#
#             Remarque: on pourra +maximiser+ la +log-vraisemblance+,
#             ce qui est numeriquement plus simple que la vraisemblance.
#
#             Une fois que cela marchera vous pourrez tenter
#             d'etendre ce travail a l'estimation jointe des
#             parametres a et b et des hyper-parametres sur
#             le bruit.
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def MinusLogLikelihood_normal(X,Y,theta,sigma=2.):
    return -np.log(likelihood_normal(X,Y,theta,sigma=sigma)) #ne resoud pas les pbs numeriques si n est grand, mais plus sympatique pour la descente de gradient

def MinusLogLikelihood_chi2(X,Y,theta,dof=3,sc=0.4):
    return -np.log(likelihood_chi2(X,Y,theta,dof=dof,sc=sc)) #ne resoud pas les pbs numeriques si n est grand, mais plus sympatique pour la descente de gradient

#3.1: fonctions pour la descente de gradient

def Grad_function(f,X,Y,theta_loc,epsilon=1e-5):
  fx=f(X,Y,theta_loc)
  p=np.size(theta_loc)
  ApproxGrad=np.zeros(p)
  veps=np.zeros(p)

  for i in range(p):
    veps[:]=0.
    veps[i]+=epsilon
    ApproxGrad[i]=(f(X,Y,theta_loc+veps)-fx)/epsilon
  return ApproxGrad

#descente de gradient avec alpha defini a la main

def grad_descent(funct,X,Y,theta_init,convspeedfactor=0.1,nbIterations=100):
    evo_f_theta=[]
    theta=theta_init.copy()
    for i in range(nbIterations):
        theta=theta-convspeedfactor*Grad_function(funct,X,Y,theta,0.001)
        evo_f_theta.append(funct(X,Y,theta))
        print(theta)


    plt.plot(evo_f_theta)
    plt.show()

    return theta

#3.2: fonctions pour calculer le maximum de vraisemblance

#utilisation:
theta_init=np.array([0.,0.])
theta_optim=grad_descent(MinusLogLikelihood_normal,X,Y,theta_init,convspeedfactor=0.01,nbIterations=1000)


likelihood_normal(X,Y,theta_init,sigma=2.,verbose=True)
likelihood_normal(X,Y,theta_optim,sigma=2.,verbose=True)

#on constate que le resultat est tres proche de ce que l'on avait avec la regression lineaire
#classique, qui minimise la somme des erreurs au carre



#utilisation:
theta_init=np.array([0.,-1.])
theta_optim=grad_descent(MinusLogLikelihood_chi2,X,Y,theta_init,convspeedfactor=0.01,nbIterations=1000)

likelihood_chi2(X,Y,theta_init,dof=3,sc=0.4,verbose=True)
likelihood_chi2(X,Y,theta_optim,dof=3,sc=0.4,verbose=True)

#l'optimisation marche aussi, meme si il faut faire tres attention au choix du theta_init (la densite du chi2
#est nulle pour les erreurs negatives)... par contre :
#  -> la vraisemblance est de 10e-7 au lieu de 10e-15
#  -> le modele lineaire colle mieux a la majorite des donnees et est moins sensible
#     aux donnees visiblement aberrantes
#
#On peut alors plus faire confiance a la pente calculee avec le bruit de type chi2 que le bruit gaussien, MAIS
#en etant clair sur le fait qu'on aura une tendance loin d'etre negligeable de s'eloigner regulierement du
#modele lineaire de maniere non-symetrique par rapport au modele.

"""
#Generation de donnees
import numpy as np
import matplotlib.pyplot as plt
n = 20
X = np.random.uniform(size=n)
y = 1.2*X[:] + np.random.chisquare(1.,size=n)*0.4-0.5

plt.plot(X[], y, 'r.')
plt.show()

MyData=np.concatenate((X.reshape(-1,1),y.reshape(-1,1)),axis=1)
np.savetxt('J18_E4_QuantifiedData.csv',MyData)
"""
