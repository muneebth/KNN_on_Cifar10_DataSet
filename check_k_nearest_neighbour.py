from data_utils import load_CIFAR10
from k_nearest_neighbour import KNearestNeighbour
import numpy as np

Xtr,Ytr,Xte,Yte=load_CIFAR10('dataset/')#loaded Cifar10 data set as training set Xtr, labels of training set as Ytr, Xte of training set,Yte of Training set 

"""Converting Image data set to Raw Date Format"""
Xtr_rows=Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
Xte_rows=Xte.reshape(Xte.shape[0],Xte.shape[1]*Xte.shape[2]*Xte.shape[3])

nn=KNearestNeighbour()
K=nn.train(Xtr_rows,Ytr)
Y_pred=np.zeros(Yte.shape[0],dtype=Ytr.dtype)
Y_pred=nn.predict(Xte_rows,K)
print "Efficiency in prediction %f for k=%d" % (np.mean(Y_pred==Yte),K)
