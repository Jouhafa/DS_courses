"""
TP5b.A : Compression d'image SVD.

"""




#1) Lecture de l'image et visualisation
import numpy as np
import matplotlib.pyplot as plt



im = plt.imread('./Brain_IRM.jpg').astype(float)


plt.imshow(im, plt.cm.gray)
plt.title('Original image')
plt.colorbar()
plt.show()


#2) Decomposition en valeurs singulieres de l'image (SVD)

[u,s,vh]=np.linalg.svd(im, full_matrices=False)

print(u.shape)
print(s.shape)
print(vh.shape)


#3) Decomposition en valeurs singulieres de l'image (SVD)


smat=np.diag(s)

im_reconstructed = np.dot(u, np.dot(smat, vh))

error=im_reconstructed-im

plt.imshow(error, plt.cm.gray)
plt.title('error')
plt.colorbar()
plt.show()



"""
Questions 1 :
-> Quantifiez l'erreur de reconstruction entre l'image reconstruite ici et l'image reconstruite
-> Vous semble-t-elle elevee ?
-> Comparez de meme le nombre de points dans l'image aux nombre de valeurs contenues dans u,s et vh. Est-ce que cette representation de l'information est interessante en l'etat ?
"""

#simple

"""
Questions 2 :
-> Reconstruisez maintenant l'image en utilisant seulement un quart des valeurs singulieres.
-> Quantifiez alors le niveau d'erreur.
-> Est-il eleve?
-> Comparez a nouveau le nombre de points dans l'image aux nombre de valeurs contenues dans la version tronquee de u,s et vh. Est-ce que cette representation compressee de l'information est interessante maintenant ?
"""

NbSV_2_keep=int(s.shape[0]/4)


u_trunk=u[:,:NbSV_2_keep]
s_trunk=s[:NbSV_2_keep]
vh_trunk=vh[:NbSV_2_keep,:]

smat_trunk=np.diag(s_trunk)

im_reconstructed = np.dot(u_trunk, np.dot(smat_trunk, vh_trunk))

plt.imshow(im_reconstructed, plt.cm.gray)
plt.title('Reconstructed image')
plt.colorbar()
plt.show()

print('Points for the original image = ', u.size+s.size+vh.size)

print('Points for the reconstructed image = ', u_trunk.size+s_trunk.size+vh_trunk.size)


error=im_reconstructed-im

plt.imshow(error, plt.cm.gray)
plt.title('error')
plt.colorbar()
plt.show()


"""
Questions 3 :
-> Reconstruisez maintenant l'image en choisissant un nombre de valeurs singulieres qui permettra de preserver environ 40 pourcents de la variabilite de l'image originale.
-> Quantifiez alors le niveau d'erreur.
-> Comparez a nouveau le nombre de points dans l'image aux nombre de valeurs contenues dans le version tronquee de u,s et vh. Conclusion ?
"""

kmax=1

while s[:kmax].sum()/s.sum() < 0.40:
  kmax+=1

print(kmax)



u_trunk=u[:,:kmax]
s_trunk=s[:kmax]
vh_trunk=vh[:kmax,:]

smat_trunk=np.diag(s_trunk)

im_reconstructed = np.dot(u_trunk, np.dot(smat_trunk, vh_trunk))

plt.imshow(im_reconstructed, plt.cm.gray)
plt.title('Reconstructed image')
plt.colorbar()
plt.show()

print('Points for the original image = ', u.size+s.size+vh.size)

print('Points for the reconstructed image = ', u_trunk.size+s_trunk.size+vh_trunk.size)


error=im_reconstructed-im

plt.imshow(error, plt.cm.gray)
plt.title('error')
plt.colorbar()
plt.show()

"""
Questions 4 :
-> Representez enfin deux courbes qui mesurent en fonction du nombre de valeurs singulieres selectionnees :
  (1) la variabilite capturee dans l'image initiale
  (2) le nombres de valeurs contenues dans le version tronquee de u,s et vh
"""

kmax_list=[]
variability_list=[]
SizeInfo_list=[]

for kmax in range(1,s.size):
  kmax_list.append(kmax)
  variability_list.append(s[:kmax].sum()/s.sum())

  u_trunk=u[:,:kmax]
  s_trunk=s[:kmax]
  vh_trunk=vh[:kmax,:]

  SizeInfo_list.append(u_trunk.size+s_trunk.size+vh_trunk.size)


kmax_arr=np.array(kmax_list)
variability_arr=np.array(variability_list)
SizeInfo_arr=np.array(SizeInfo_list)



plt.plot(kmax_arr,variability_arr)
plt.title('Variabilite capturee par rapport au nb de valeur singulieres')
plt.show()

ref_size=u.size+s.size+vh.size
plt.plot(kmax_arr,SizeInfo_arr)
plt.title('Info necessaire par rapport au nb de valeur singulieres\n ('+str(ref_size)+' dans l\'image originale)')
plt.show()
