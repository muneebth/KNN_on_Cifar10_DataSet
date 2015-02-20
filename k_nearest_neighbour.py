import numpy as np

class KNearestNeighbour:

   def __init__(self):
       self.K=0
       
   
   def train(self,X,Y):
       """Input X :Training Set features
                Y :Labels of Training Set""" 
       val_eff=np.zeros((5,10),dtype=np.float64)  #declared variable for calculating efficiencies in  5 fold cross validation set 
       for i in range(1,6):   
          """5 fold Cross validation starts from here """
          num_train=X.shape[0]     #size of training set is calculated
          if i==1:   
             """ Training set is made as other than first block """       
             self.X_tr=X[i*(num_train/5):,:]  
             self.Y_tr=Y[i*(num_train/5):]
          else:
             """Training set is made appending the block other than validation set"""
             self.X_tr=np.append(X[:(i-1)*(num_train/5),:], X[i*(num_train/5):,:],0)  
             self.Y_tr=np.append(Y[:(i-1)*(num_train/5)], Y[i*(num_train/5):],0)
          self.X_val=X[(i-1)*(num_train/5):i*(num_train/5),:] # Validation set is build
          self.Y_val=Y[(i-1)*(num_train/5):i*(num_train/5)]   
          self.K=1
         
          Y_pred=np.zeros((self.X_val.shape[0]),dtype=self.Y_val.dtype)
              
          print 'Size of Validation set ',self.X_val.shape
          print 'Size of Training set ',self.X_tr.shape
          for j in range(1,11):
             """Calculation of  efficiencies for different values of Hyperparameter K"""
             Y_pred=self.predict(self.X_val,j)
             val_eff[(i-1),(j-1)]=np.mean(self.Y_val==Y_pred)
             print "Efficiency of validation set : %f  when block is %d  k is %d" % (np.mean(Y_pred==self.Y_val),i,j)       
       
       print "Validation set ",val_eff
       print "Mean of validation set ",np.mean(val_eff,0)
       self.K=(np.argmax(np.mean(val_eff,0)))+1
       print "Maximum efficiency  is for K=%d "%(self.K)
       
       return self.K
       
       
       
   def predict(self,X,k=1):
       """In this function we find the label for Input :X  based on the distance calculated """
       
       num_test=X.shape[0]
       dists=np.zeros((num_test,self.X_tr.shape[0]),dtype=self.X_tr.dtype)
       
       for i in xrange(num_test):
           """Distance of test sample  i from all training set is calculated """ 
           dists[i,:]=np.sqrt(np.sum(np.square(self.X_tr-X[i,:]),1))
       
       
       return self.predict_label(dists,k)
       
             
   def predict_label(self,dists,k):
       """In this function we Predict the label based on the maximum occurence of the  top K closest set labels in the training set """   
       num_test=dists.shape[0]
       Y_pr=np.zeros(num_test)
       for i in xrange(num_test):
           nearest_tag=[]
           nearest_index=np.argsort(dists[i,:])[:k]
           #print " Nearest Tag ",self.Y_tr[nearest_index]
           nearest_tag=self.Y_tr[nearest_index].tolist()
           Y_pr[i]=max(nearest_tag,key=nearest_tag.count)
       
       return Y_pr
