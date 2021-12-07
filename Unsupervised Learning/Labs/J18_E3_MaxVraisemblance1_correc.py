
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Manipulation du maximum de vraisemblance :
#
#
#    Vérifier empiriquement comment évolue ce maximum de vraisemblance si l'on effectue de plus en plus de tirages
#    Que se passe-t-il quand il y a trop de tirages ? Représenter la log-vraisemblance plutot que la vraisemblance dans ce cas.

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import matplotlib.pyplot as plt


#QUESTION 1:
#Tirer 10 fois une pièce à pile ou face et modéliser les résultats obtenus comme ceux d'une variable aléatoire X qui
#vaut X_i=0 si on a pile et X_i=1 si on a face.
#
#Calculez le maximum de vraisemblance du paramètre p d'un loi de Bernoulli qui modéliserait le problème. Pour y arriver,
#différentes valeures possibles de p seront testées et le p retenu sera celui qui a la plus grande vraisemblance.

NbTirages=10
NbPiles = 5
NbFaces=NbTirages-NbPiles

PossibleValuesForP=np.linspace(0.01,0.99,100)


def vraisemblance(n_pile, n_face, p):
    return p**n_pile * (1-p)**n_face

CorrespondingLikelihood = [vraisemblance(NbPiles, NbFaces, p) for p in PossibleValuesForP]

plt.plot(PossibleValuesForP,CorrespondingLikelihood)
plt.show()


#QUESTION 2:
#En codant un simulateur de pile ou face, observez comme evolue la vraisemblance pour 5, 10, 20, 50, 200, 1000, 10000 lancers
#(on pourra utiliser la commande : " NbPiles = np.sum(np.random.randn(NbTirages) > 0) " )


for NbTirages in [5, 10, 20, 50, 200, 1000, 10000]:
    NbPiles = np.sum(np.random.randn(NbTirages) > 0)
    NbFaces=NbTirages-NbPiles

    CorrespondingLikelihood = [vraisemblance(NbPiles, NbFaces, p) for p in PossibleValuesForP]

    plt.plot(PossibleValuesForP,CorrespondingLikelihood)
    plt.title(str(NbTirages)+' tirages')
    plt.show()


#QUESTION 3:
#On remarquera que l'on a des problemes numériques quand le nombre d'observations est trop grand. Reproduisez le test
#en utilisant la log-vraisemblance au lieu de la vraisemblance. Lorsque le nombre d'observations est raisonable, on
#constatera que la log-vraisemblance et la vraisemblance on leur maximum au meme point

def log_vraisemblance(n_pile, n_face, p):
    return n_pile*np.log(p) + n_face*np.log(1-p)


for NbTirages in [5, 10, 20, 50, 200, 1000, 10000]:
    NbPiles = np.sum(np.random.randn(NbTirages) > 0)
    NbFaces=NbTirages-NbPiles

    CorrespondingLogLikelihood = [log_vraisemblance(NbPiles, NbFaces, p) for p in PossibleValuesForP]
    CorrespondingLikelihood = [vraisemblance(NbPiles, NbFaces, p) for p in PossibleValuesForP]

    plt.plot(PossibleValuesForP,CorrespondingLogLikelihood)
    plt.title(str(NbTirages)+' tirages (LogLikelihood)')
    plt.show()

    plt.plot(PossibleValuesForP,CorrespondingLikelihood)
    plt.title(str(NbTirages)+' tirages (Likelihood)')
    plt.show()
