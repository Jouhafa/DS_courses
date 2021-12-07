
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#TP1_Exo2) segmentation d'image couleur avec l'algorithme des K-means
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np                 #pour faire des mathematiques numeriques
import matplotlib.pyplot as plt    #pour afficher des images (fonction imshow)
import numpy.random as nprand      #pour tirer des valeurs aléatoirement
import scipy.ndimage as scim       #pour tout ce qui tourne autour du filtrage et le traitement d'image de base
import matplotlib.pyplot


#-----------------------------------------------------------------------
#1) fonctions de visualisation des images
#-----------------------------------------------------------------------

def showImage3channels(InputImage):
  plt.figure()
  plt.imshow(InputImage[:,:,0],cmap='Greys')
  plt.title('channel 1: red')
  plt.colorbar()

  plt.figure()
  plt.imshow(InputImage[:,:,1],cmap='Greys')
  plt.title('channel 2: green')
  plt.colorbar()

  plt.figure()
  plt.imshow(InputImage[:,:,2],cmap='Greys')
  plt.title('channel 3: blue')
  plt.colorbar()

  plt.show()


def showImage1channel(InputImage):
  imgplot = plt.imshow(InputImage,cmap='Greys')
  #imgplot = plt.imshow(InputImage)
  plt.colorbar()
  plt.show()


#-----------------------------------------------------------------------
#2) segmentation automatique d'image couleur avec l'algorithme des K-means
#-----------------------------------------------------------------------


Im2=matplotlib.pyplot.imread('J18_E1_kmeans.jpg')
Im=Im2*1.               #lecture de l'image 2D+[3 canaux]  -> Im.shape=[[nb de pixels en x],[nb de pixels en y],[nb de canaux]]
showImage3channels(Im)


#QUESTION 1 : retransformer la forme de l'image pour qu'elle soit un array
#2D de taille (M,3), ou M est le nombre de pixel dans l'image et 3
#correspond au cannaux RGB dans l'image. Chaque pixel de l'image peut
#alors etre considere comme une observation en dimension 3.


VectorisedImage=Im.reshape((Im.shape[0]*Im.shape[1],Im.shape[2]))   #Vectorise Im de sorte a ce qu'il soit de taille [[nb de pixels en x]*[nb de pixels en y],[nb de canaux]].
                                                                    #Il sera alors considere comme [nb de pixels en x]*[nb de pixels en y] observations de dimension [nb de canaux]
                                                                    #par un algorithme de clustering (ici les k-means).
print(Im[0,0:10,:])
print(VectorisedImage[0:10,:])

print(Im[1,0:10,:])
print(VectorisedImage[Im.shape[1]+0:Im.shape[1]+10,:])


#QUESTION2 : Utiliser l'algorithme de K-means pour attribuer un label (segmenter)
#a chaque pixel de l'image

import scipy.cluster.vq as scipyvq


k=4         #nombre de clusters de l'algorithme de kmeans
[VectorisedImageCenters,VectorisedImageSegmentation]=scipyvq.kmeans2(VectorisedImage, k, iter=100)  #lancement de l'algorithme de clustering.
                                                                                                    #Taper "scipyvq.kmeans2?" dans l'editeur de commandes pour bien comprendre les deux outputs.

#QUESTION 3 : transformer la forme des labels pour quelle corresponde a la forme
#de l'image initiale

ImageSegmentation = VectorisedImageSegmentation.reshape((Im.shape[0],Im.shape[1]))      #transforme les labels 'vectorises' sous la forme de l'image originale ([nb de pixels en x]*[nb de pixels en y])

showImage1channel(ImageSegmentation)







#-----------------------------------------------------------------------
#3) POUR INFO : CORRECTION SANS UTILISER LA FONCTION SCIPY
#-----------------------------------------------------------------------


#3.1) init

#3.1.1) generate a grey level image

Im2=matplotlib.pyplot.imread('voiture.jpg')
Im=Im2*1.
showImage3channels(Im)



#3.1.2) randomly generate grey level centers in each group of data
K=5  #number of classes
MinInt=np.array([Im[:,:,0].min(),Im[:,:,1].min(),Im[:,:,2].min()])
MaxInt=np.array([Im[:,:,0].max(),Im[:,:,1].max(),Im[:,:,2].max()])
GLmeans=MinInt+nprand.rand(K,3)*(MaxInt-MinInt)
GLsize=np.zeros(K)

#3.1.3) allocate memory for the segmentation image
SegImage=np.zeros((Im.shape[0],Im.shape[1]), dtype='uint8')

print 'init:'
print GLmeans

#3.2) iterative segmentation algorithm
changesNumber=Im.shape[1]*Im.shape[0]
iteration=0
while changesNumber>5*Im.shape[1]*Im.shape[0]/100:
  changesNumber=0
  iteration=iteration+1
  #3.2.1) find the class center which has the closest intensity to Im[i,j]
  for i in range(Im.shape[0]):
    for j in range(Im.shape[1]):
      formerClass=SegImage[i,j]
      locClass=0
      SegImage[i,j]=locClass
      #minDistance=np.max(np.abs(GLmeans[locClass,:]-Im[i,j,:]))
      minDistance=np.sum((GLmeans[locClass,:]-Im[i,j,:])*(GLmeans[locClass,:]-Im[i,j,:]))
      for locClass in range(K):
        #tmpDist=np.max(np.abs(GLmeans[locClass,:]-Im[i,j,:]))
        tmpDist=np.sum((GLmeans[locClass,:]-Im[i,j,:])*(GLmeans[locClass,:]-Im[i,j,:]))
        if tmpDist<minDistance:
          SegImage[i,j]=locClass
          minDistance=tmpDist
      if formerClass!=SegImage[i,j]:
        changesNumber=changesNumber+1

  #3.2.2) update class centers
  GLmeans=GLmeans*0.
  GLsize=GLsize*0.
  for i in range(Im.shape[0]):
    for j in range(Im.shape[1]):
      GLmeans[SegImage[i,j],:]=GLmeans[SegImage[i,j],:]+Im[i,j,:]
      GLsize[SegImage[i,j]]=GLsize[SegImage[i,j]]+1.

  for i in range(K):
    if GLsize[i]<0.5:
      print "Class "+str(i)+" was void -> new centers generated"
      GLmeans[i,:]=MinInt+nprand.rand(3)*(MaxInt-MinInt)
    else:
      GLmeans[i,:]=GLmeans[i,:]/GLsize[i]

  print 'iteration '+str(iteration)+': '+str(changesNumber)+' voxels with changed class  / thresh='+str(2*Im.shape[1]*Im.shape[0]/100)
  print GLmeans


imgplot = plt.imshow(SegImage)
plt.colorbar()
plt.show()
