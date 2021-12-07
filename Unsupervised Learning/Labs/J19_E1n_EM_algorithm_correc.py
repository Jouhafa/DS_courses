
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Algorithme E.M. pour le melange de Gaussiennes :
#
#On observe des observations en 2D qui semblent appartenir a trois sous groupes
#differents. Nous allons utiliser un modele Gaussien pour representer la
#moyenne et le niveau d'incertitude associe a chaque groupe et un algorithme
#E.M. pour clusteriser les observations et trouver les proprietes des
#Gaussiennes.

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import matplotlib.pyplot as plt

X=np.genfromtxt('J19_E1n_QuantifiedData.csv')

plt.scatter(X[:,0],X[:,1])
plt.show()


#QUESTION 1: Clusterisez les donnes avec un algorithme de K-means avec trois
#groupes.


from sklearn.cluster import KMeans

KMclassifier=KMeans(n_clusters=3)

KMclassifier.fit(X)

cluster_Labels=KMclassifier.labels_


plt.scatter(X[:,0],X[:,1],c=cluster_Labels[:],cmap='rainbow',alpha=0.5)
plt.colorbar()
plt.show()

#On constatera que les deux groupe detectes en haut de la figure semblent
#s'etendre vers le centre de la figure avec des observations peu denses.
#On se dit qu'il est alors fort possible que les trois groupes soient
#echantillonnes avec des lois gaussiennes ayant des proprietes differentes, ce
#qui conduit a avoir de mauvais resultats avec les K-means.
#Nous allons ainsi resoudre le probleme de clustering a l'aide d'un algorithme
#E.M. qui estime a la fois les clusters, et les proprietes de la distribution
#des observations dans chaque groupe.

#Pour modeliser le probleme, nous allons reproduire le modele du cours.
#Pour simplifier le probleme, on considerera que les variances des lois dans
#chaque classe seront isotropes, c'est a dire que les matrices
#Sigma_j = [ [sigma_j, 0.] , [0. , sigma_j] ] , ou sigma_j est un scalaire.
#
#Les parametres a estimer seront alors :
#  theta=(tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3)
#avec tau_3=1.-tau_1-tau_2 (probabilite globale d'avoir des observations dans chaque
#groupe)

#QUESTION 2: Quelle est la log-vraisemblance du modele complet (avec les X et
#les Z)? On se souviendra que
#    $log ( \Pi_{i=1}^{n} ( a_i ) )   =  \sum_{i=1}^{n} ( log (a_i) ) $
#
#Remarque : On pourra utiliser scipy.stats.multivariate_normal.pdf pour mesurer
#la densite de probabilite d'une loi normale multivariee (ici 2D).
#
#Remarque : Pour estimer la vraissemblance du modele complet, on pourra tirer
#pour l'instant au hazard (dans 0, 1, 2) les labels Z associes a chaque
#observation.


from scipy.stats import multivariate_normal

def log_likelihood(X,theta,Z):

    [n,p]=X.shape
    [tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]=theta
    tau_3=1.-tau_1-tau_2

    log_likelihood=0.

    for locGroup in [0,1,2]:
        #define the parameters related to this group
        if locGroup==0:
            tau=tau_1
            mu_x=mu_1_x
            mu_y=mu_1_y
            sigma=sigma_1
        elif locGroup==1:
            tau=tau_2
            mu_x=mu_2_x
            mu_y=mu_2_y
            sigma=sigma_2
        else:
            tau=tau_3
            mu_x=mu_3_x
            mu_y=mu_3_y
            sigma=sigma_3

        #compute the log-likelihood in this group
        for i in range(n):
            mean_i=np.array([mu_x,mu_y])
            cov_i=np.array([[sigma,0.],[0.,sigma]])
            if Z[i]==locGroup:
                log_likelihood+=np.log(tau*multivariate_normal.pdf(X[i,:], mean=mean_i, cov=cov_i))

    return log_likelihood


#QUESTION 3: Testez cette fonction pour trouver manuellement une parametrisation
#            qui vous semble raisonable. La log-vraissemblance est-elle bien
#            meilleur pour une parametrisation theta qui vous semble raisonable
#            que pour d'autres qui ont peu de sens?

#-> raisonable
tau_1=0.33 ; tau_2=0.33 ;
mu_1_x=-1.8 ; mu_1_y=1.8 ; sigma_1=0.5
mu_2_x=1.8 ; mu_2_y=1.8 ; sigma_2=0.5
mu_3_x=0 ; mu_3_y=-0.2 ; sigma_3=1.2
theta=[tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]
Z=np.random.randint(low=0,high=3,size=X.shape[0])

print(log_likelihood(X,theta,Z))

#-> faux
tau_1=0.33 ; tau_2=0.33
mu_1_x=-2.5 ; mu_1_y=2.5 ; sigma_1=0.8
mu_2_x=2.5 ; mu_2_y=2.5 ; sigma_2=0.8
mu_3_x=0 ;  mu_3_y=-2 ; sigma_3=1.
theta=[tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]
Z=np.random.randint(low=0,high=3,size=X.shape[0])

print(log_likelihood(X,theta,Z))

#-> vraiment faux
tau_1=0.33 ; tau_2=0.33
mu_1_x=-3 ; mu_1_y=3 ; sigma_1=0.5
mu_2_x=3 ; mu_2_y=3 ; sigma_2=0.5
mu_3_x=0 ; mu_3_y=-4 ; sigma_3=0.2
theta=[tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]
Z=np.random.randint(low=0,high=3,size=X.shape[0])

print(log_likelihood(X,theta,Z))

#-> comportement qui semble correct

#QUESTION 4: - Faire trois figures qui montrent la probabilite estimee pour la parametrisation
#              theta que chaque observation soit dans la classe 1, 2 ou 3. On se souviendra que
#              ces trois probabilites seront utilises pour evaluer l'esperance du modele
#              optimise dans l'algorithme EM.
#            - Montrer alors la classe la plus probable pour a chaque observation pour une
#              parametrisation theta donnee
#            - Observer ces figures pour plusieurs parametrisations theta


def cpt_probabilities(X,theta,showProbabilities=True,showClusters=True):

    [n,p]=X.shape
    [tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]=theta
    tau_3=1.-tau_1-tau_2

    #compute the densities
    densities=[]

    for locGroup in [0,1,2]:
        #define the parameters related to this group
        if locGroup==0:
            tau=tau_1
            mu_x=mu_1_x
            mu_y=mu_1_y
            sigma=sigma_1
        elif locGroup==1:
            tau=tau_2
            mu_x=mu_2_x
            mu_y=mu_2_y
            sigma=sigma_2
        else:
            tau=tau_3
            mu_x=mu_3_x
            mu_y=mu_3_y
            sigma=sigma_3

        densities.append([])
        for i in range(n):
            mean_i=np.array([mu_x,mu_y])
            cov_i=np.array([[sigma,0.],[0.,sigma]])
            #print(X[i,:],mean_i,cov_i)
            densities[-1].append(tau*multivariate_normal.pdf(X[i,:], mean=mean_i, cov=cov_i))

    #compute the probabilities
    probabilities=[]
    probabilities.append([])
    probabilities.append([])
    probabilities.append([])

    for i in range(n):
        probabilities[0].append(densities[0][i]/(densities[0][i]+densities[1][i]+densities[2][i]))
        probabilities[1].append(densities[1][i]/(densities[0][i]+densities[1][i]+densities[2][i]))
        probabilities[2].append(densities[2][i]/(densities[0][i]+densities[1][i]+densities[2][i]))

    if showProbabilities:
        for locGroup in [0,1,2]:
            #plt.scatter(X[:,0],X[:,1],c=densities[locGroup],cmap='rainbow',alpha=0.5)
            plt.scatter(X[:,0],X[:,1],c=probabilities[locGroup],cmap='rainbow',alpha=0.5)
            plt.title('probabilities of each group '+str(locGroup))
            plt.colorbar()
            plt.show()

    #compute the max density -> clustering
    labels=[]
    for i in range(len(probabilities[0])):
        label_i=0
        if probabilities[1][i]>probabilities[0][i] and probabilities[1][i]>probabilities[2][i]:
            label_i=1
        if probabilities[2][i]>probabilities[0][i] and probabilities[2][i]>probabilities[1][i]:
            label_i=2
        labels.append(label_i)

    if showClusters:
        plt.scatter(X[:,0],X[:,1],c=labels,cmap='rainbow',alpha=0.5)
        plt.title('Labels')
        plt.colorbar()
        plt.show()

    return [probabilities,labels]


#-> raisonable

tau_1=0.33 ; tau_2=0.33 ;
mu_1_x=-1.8 ; mu_1_y=1.8 ; sigma_1=0.3
mu_2_x=1.8 ; mu_2_y=1.8 ; sigma_2=0.3
mu_3_x=0 ; mu_3_y=-1.2 ; sigma_3=1.2
theta=[tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]

cpt_probabilities(X,theta)

#-> faux
tau_1=0.33 ; tau_2=0.33
mu_1_x=-2.5 ; mu_1_y=2.5 ; sigma_1=0.8
mu_2_x=2.5 ; mu_2_y=2.5 ; sigma_2=0.8
mu_3_x=0 ;  mu_3_y=-2 ; sigma_3=1.
theta=[tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]

cpt_probabilities(X,theta)


#QUESTION 5: Nous allons maintenant coder la fonction d'esperance de la vraisemblance
#            dans laquelle les labels Z_i sont remplaces par leur probabilite. Cette
#            fonction sera maximisee dans l'etape "M" de l'algorithme E.M.

def funct_Q(X,theta,probabilities):

    [n,p]=X.shape
    [tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]=theta
    tau_3=1.-tau_1-tau_2

    Q_score=0.

    for locGroup in [0,1,2]:
        #define the parameters related to this group
        if locGroup==0:
            tau=tau_1
            mu_x=mu_1_x
            mu_y=mu_1_y
            sigma=sigma_1
        elif locGroup==1:
            tau=tau_2
            mu_x=mu_2_x
            mu_y=mu_2_y
            sigma=sigma_2
        else:
            tau=tau_3
            mu_x=mu_3_x
            mu_y=mu_3_y
            sigma=sigma_3

        #compute the log-likelihood in this group
        for i in range(n):
            mean_i=np.array([mu_x,mu_y])
            cov_i=np.array([[sigma,0.],[0.,sigma]])
            #print('->',X[i,:],mean_i,cov_i)
            Q_score+=probabilities[locGroup][i]*np.log(tau*multivariate_normal.pdf(X[i,:], mean=mean_i, cov=cov_i))

    return Q_score

#QUESTION 6: Nous allons maintenant resoudre l'algorithme E.M. Pour y arriver, deux options
#            sont possibles lors de l'etape "M":
#            - Option 1 : Vous maximisez la fonction codee dans la question precedente avec
#                         un algorithme de descente de gradient. Si vous partez sur cette
#                         option, il faudra bien remarquer que la derivee de l'esperance de
#                         la log-vraisemblance par rapport aux \mu_k n'est pas forcement sur
#                         la meme echelle que celle par rapport aux \Sigma_k ou bien aux \tau_k.
#                         Avant d'optimiser tous ces parametres a la fois, commencez alors par
#                         n'optimiser que les parametres \mu_k.
#            - Option 2 : Vous pouvez aussi analytiquement calculer la derivee de l'esperance de
#                         la log-vraisemblance par rapport a chacun des termes a optimiser (i.e
#                         calculer son gradient) et considerer que le maximum de la fonction est
#                         obtenu pour les zeros du gradient (on a une fonction convexe).


# -> option 1

def Grad_funct_Q_wrt_theta(X,theta_loc,probabilities,epsilon=1e-3,dimensions=[-1]):
    """
    Si le premier element de dimensions est -1 alors le gradient sera calcule sur
    toutes les dimensions de theta_loc. Il le sera sinon seulement sur les dimensions
    specifiees dans la liste
    """
    if dimensions[0]==-1:
        dimensions=range(p)

    fx=funct_Q(X,theta_loc,probabilities)
    p=np.size(theta_loc)
    ApproxGrad=np.zeros(p)
    veps=np.zeros(p)
    for i in dimensions:
        veps[:]=0.
        veps[i]+=epsilon
        ApproxGrad[i]=(funct_Q(X,theta_loc+veps,probabilities)-fx)/epsilon

    return ApproxGrad




tau_1=0.33 ; tau_2=0.33 ;
mu_1_x=-1.8 ; mu_1_y=1.8 ; sigma_1=0.3
mu_2_x=1.8 ; mu_2_y=1.8 ; sigma_2=0.3
mu_3_x=0 ; mu_3_y=-1.2 ; sigma_3=1.2
theta_init=[tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]

theta=theta_init.copy()

print('theta_init: '+str(np.round(theta[:2],2))+' | '+str(np.round(theta[2:4],2))+'  '+str(np.round(theta[4:6],2))+'  '+str(np.round(theta[6:8],2))+'  '+str(np.round(theta[8:],2)))
list_L=[]

#optimize on the centroids
nb_GD_Mstep_it=5
convspeedfactor_GD_Mstep=0.001
dims=[2,3,4,5,6,7]
nb_EM_it=20

for it in range(nb_EM_it):
    print('iteration '+str(it))
    #E-step
    [probabilities,Z]=cpt_probabilities(X,theta,showProbabilities=False,showClusters=False)

    #M-step (grad-descent)
    evo_f_theta=[]
    for i in range(nb_GD_Mstep_it):
        theta=theta+convspeedfactor_GD_Mstep*Grad_funct_Q_wrt_theta(X,theta,probabilities,dimensions=dims) # + is because we maximise Q
        evo_f_theta.append(funct_Q(X,theta,probabilities))
        print(str(np.round(theta[:2],2))+' | '+str(np.round(theta[2:4],2))+'  '+str(np.round(theta[4:6],2))+'  '+str(np.round(theta[6:8],2))+'  '+str(np.round(theta[8:],2)))
        list_L.append(log_likelihood(X,theta,Z))



plt.plot(list_L)
plt.show()


cpt_probabilities(X,theta_init,showProbabilities=False,showClusters=True)

cpt_probabilities(X,theta,showProbabilities=False,showClusters=True)


# -> option 2
#On a :
#  (1) Par definition : $p_k = \frac{1}{n} \tilde_{p}_{i,k}$
#  (2) $\frac{\partial \mathbb{E}[log(L(...))]{\partial \mu_k^l} = 0$
#      donc $\mu_k^l = \frac{\sum_{i=1}^n ( \tilde{p}_{k,i} x_i^l ) }{\sum_{i=1}^n ( \tilde{p}_{k,i} )}$
#  (3) $\frac{\partial \mathbb{E}[log(L(...))]{\partial \sigma_k} = 0$
#      donc $\sigma_k = \frac{ \sum_{i=1}^n [ \tilde{p}_{k,i}  \sum_{l=1}^p (x_i^l - \mu_i^l)^2   ]  }{p \sum_{i=1}^n \tilde{p}_{k,i}}$


def M_step_direct(X,theta_loc,probabilities):
    [n,p]=X.shape
    [tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]=theta_loc
    tau_3=1.-tau_1-tau_2

    theta_new=theta_loc

    probas_class1=np.array(probabilities[0])
    probas_class2=np.array(probabilities[1])
    probas_class3=np.array(probabilities[2])

    #update tau_1 and tau_2
    theta_new[0]=np.mean(probas_class1) # tau_1
    theta_new[1]=np.mean(probas_class2) # tau_2


    #update mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y
    theta_new[2]= np.sum(probas_class1*X[:,0].ravel()) / (np.sum(probas_class1)) #mu_1_x
    theta_new[3]= np.sum(probas_class1*X[:,1].ravel()) / (np.sum(probas_class1)) #mu_1_y
    theta_new[4]= np.sum(probas_class2*X[:,0].ravel()) / (np.sum(probas_class2)) #mu_2_x
    theta_new[5]= np.sum(probas_class2*X[:,1].ravel()) / (np.sum(probas_class2)) #mu_2_y
    theta_new[6]= np.sum(probas_class3*X[:,0].ravel()) / (np.sum(probas_class3)) #mu_3_x
    theta_new[7]= np.sum(probas_class3*X[:,1].ravel()) / (np.sum(probas_class3)) #mu_3_y

    #update sigma_1,sigma_2,sigma_3
    tmp=np.power( X[:,0].ravel()-mu_1_x ,2.) + np.power( X[:,1].ravel()-mu_1_y ,2.)
    theta_new[8]= np.sum(probas_class1*tmp) / (2*np.sum(probas_class1)) #sigma_1
    tmp=np.power( X[:,0].ravel()-mu_2_x ,2.) + np.power( X[:,1].ravel()-mu_2_y ,2.)
    theta_new[9]= np.sum(probas_class2*tmp) / (2*np.sum(probas_class2)) #sigma_2
    tmp=np.power( X[:,0].ravel()-mu_3_x ,2.) + np.power( X[:,1].ravel()-mu_3_y ,2.)
    theta_new[10]= np.sum(probas_class3*tmp) / (2*np.sum(probas_class3)) #sigma_3

    return theta_new


tau_1=0.33 ; tau_2=0.33 ;
mu_1_x=-1.8 ; mu_1_y=1.8 ; sigma_1=0.3
mu_2_x=1.8 ; mu_2_y=1.8 ; sigma_2=0.3
mu_3_x=0 ; mu_3_y=-1.2 ; sigma_3=1.2
theta_init=[tau_1,tau_2,mu_1_x,mu_1_y,mu_2_x,mu_2_y,mu_3_x,mu_3_y,sigma_1,sigma_2,sigma_3]

theta=theta_init.copy()

print('theta init: '+str(np.round(theta[:2],2))+' | '+str(np.round(theta[2:4],2))+'  '+str(np.round(theta[4:6],2))+'  '+str(np.round(theta[6:8],2))+'  '+str(np.round(theta[8:],2)))

#optimize on the centroids
nb_EM_it=15

list_L=[]
for it in range(nb_EM_it):
    print('iteration '+str(it))
    #E-step
    [probabilities,Z]=cpt_probabilities(X,theta,showProbabilities=False,showClusters=False)

    #M-step (grad-descent)
    theta=M_step_direct(X,theta,probabilities)

    print('theta: '+str(np.round(theta[:2],2))+' | '+str(np.round(theta[2:4],2))+'  '+str(np.round(theta[4:6],2))+'  '+str(np.round(theta[6:8],2))+'  '+str(np.round(theta[8:],2)))
    list_L.append(log_likelihood(X,theta,Z))


plt.plot(list_L)
plt.show()


cpt_probabilities(X,theta_init,showProbabilities=False,showClusters=True)

cpt_probabilities(X,theta,showProbabilities=False,showClusters=True)





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


"""
#Generation de donnees

from scipy.stats import multivariate_normal

theta=np.pi/3
rotmat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
scalingmat=np.array([[1.,0],[0.,1.]])

group1=multivariate_normal.rvs([0.,0.], np.dot(scalingmat,rotmat),size=50)
group2=multivariate_normal.rvs([2,2.0], [[0.2,0.],[0.,0.2]],size=50)
group3=multivariate_normal.rvs([-2.0,2], [[0.1,0.],[0.,0.1]],size=50)

plt.scatter(group1[:,0],group1[:,1])
plt.scatter(group2[:,0],group2[:,1])
plt.scatter(group3[:,0],group3[:,1])
plt.show()

X=np.concatenate((group1,group2,group3))
np.savetxt('J19_E1n_QuantifiedData.csv',X)
"""

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
