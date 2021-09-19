import numpy as np
import tensorflow as tf
import random
import time
import json
from PIL import Image
import cupy as cp

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
data=None

class NueralNetwork:


    def getrandparameters(self,row,col):
        return cp.random.randn(row,col)*cp.sqrt(1/(col))
    def __init__(self,sizes,isTrained=False):

        #input layer
        self.flatten=None 
       
        #layer 1
        self.layer1_weights=cp.array(data["0"]) if isTrained else self.getrandparameters(sizes[1],sizes[0])
        self.layer1_bias=cp.array(data["1"]) if isTrained else self.getrandparameters(sizes[1],1)
        
        #layer 2
        self.layer2_weights=cp.array(data["2"]) if isTrained else self.getrandparameters(sizes[2],sizes[1])
        self.layer2_bias=cp.array(data["3"]) if isTrained else self.getrandparameters(sizes[2],1)
       
        #layer3
        self.layer3_weights=cp.array(data["4"]) if isTrained else self.getrandparameters(sizes[3],sizes[2])
        self.layer3_bias=cp.array(data["5"]) if isTrained else self.getrandparameters(sizes[3],1)
        
       
      

        #loss
        self.loss=0
        self.loss_test=0
        
        if isTrained:
            print('Parameters Loaded')
        else:
            print('Parameters Initialized')
        
    def ReLU(self,x):
        ans=x*(x>0)
        x=None
        return ans 
    def ReLUd(self,x):
       
        return (x>0)*1
    def softmax(self,x):
    
        x2=cp.max(x,axis=1).reshape(x.shape[0],1,1)
    
        
        e_x = cp.exp(x - x2)
        
        e_sum=cp.sum(e_x,axis=1).reshape(x.shape[0],1,1)
    
        return cp.divide(e_x ,e_sum)
    
    def run(self,img,p,training):
        
       
        
        #Flatten the image
        batch_size,h,w=img.shape
        
        self.flatten=img.reshape((batch_size,h*w,1))

        curr_layer=self.flatten
     
        #layer 1
        
        self.layer1=self.ReLU(cp.matmul(self.layer1_weights,curr_layer)+self.layer1_bias)
        
        if training:
            self.layer1_dropout=cp.random.binomial(1, p, size=self.layer1.shape) / p
            self.layer1*=self.layer1_dropout
        curr_layer=self.layer1
        
        #layer 2
        self.layer2=self.ReLU(cp.matmul(self.layer2_weights,curr_layer)+self.layer2_bias)
        if training:
            self.layer2_dropout=cp.random.binomial(1, p, size=self.layer2.shape) / p
            self.layer2*=self.layer2_dropout
        curr_layer=self.layer2
        
        #layer3
       
        self.layer3=self.softmax(cp.matmul(self.layer3_weights,curr_layer)+self.layer3_bias)
        
        last_val=self.layer3.reshape(batch_size,10,)
        curr_layer=None
        return cp.argmax(last_val,axis=1)
    def findgradient(self,truth,training,num_images,s):
        #update loss
        # layer3_list=self.layer3.tolist()
        # for i in range(len(truth)):
        #     if truth[i]:
               
        #         val=-1*np.log(layer3_list[i][0])/num_images 
        #         self.loss+=val if val!=float('inf') else 0 
        #     else:
                
              
                
              
        #         val=-1*np.log(1-layer3_list[i][0])/num_images
        #         self.loss+=val if val!=float('inf') else 0
                            

        # self.loss+=(np.sum((self.layer1_weights)**2)+np.sum((self.layer2_weights)**2)+np.sum((self.layer3_weights)**2))*(s/(2*num_images))
        # truth=truth.reshape(truth.shape[0],1)
      
        truth=truth.reshape(truth.shape[0],10,1)
       
        curr_error=self.layer3-truth
    


        #layer 3
        
        # if first_img:
        
        self.layer3_weights_gradient=cp.matmul(curr_error,self.layer2.transpose(0,2,1))
        
        self.layer3_bias_gradient=curr_error
        
        # else:
        #     self.layer3_weights_gradient+=curr_error.dot(np.transpose(self.layer2))
        #     self.layer3_bias_gradient+=curr_error
      
        curr_error=  cp.matmul(self.layer3_weights.T,curr_error)
       
        #layer 2
        
        curr_error=curr_error*self.ReLUd(self.layer2)
        if training:
            curr_error*=self.layer2_dropout
        # if first_img:

        #     self.layer2_weights_gradient=curr_error.dot(np.transpose(self.layer1))
        #     self.layer2_bias_gradient=curr_error
        # else:
        
        self.layer2_weights_gradient=cp.matmul(curr_error,self.layer1.transpose(0,2,1))

        self.layer2_bias_gradient=curr_error
       
        curr_error= np.matmul(self.layer2_weights.T,curr_error)
       
        #layer 1 
        curr_error=curr_error*self.ReLUd(self.layer1)
        if training:
            curr_error*=self.layer1_dropout
        # if first_img:
            
        #     self.layer1_weights_gradient=curr_error.dot(np.transpose(self.flatten))
        #     self.layer1_bias_gradient=curr_error
        # else:
        
        self.layer1_weights_gradient=cp.matmul(curr_error,self.flatten.transpose(0,2,1))
        self.layer1_bias_gradient=curr_error
        
        
        
    def gradient_descent(self,n,num_images,s,batch_size):
        
        #layer 1
        # self.layer1_weights*(1-(s*n)/num_images)
        self.layer1_weights-=n*cp.sum(self.layer1_weights_gradient,axis=0)*(1/batch_size)
        
        
        
        self.layer1_bias-=n*cp.sum(self.layer1_bias_gradient,axis=0)*(1/batch_size)

        #layer 2
        # self.layer2_weights*(1-(s*n)/num_images)
        self.layer2_weights-=n*cp.sum(self.layer2_weights_gradient,axis=0)*(1/batch_size)
      
        self.layer2_bias-=n*cp.sum(self.layer2_bias_gradient,axis=0)*(1/batch_size)
       
        #layer 3
        # self.layer3_weights*(1-(s*n)/num_images)
        self.layer3_weights-=n*cp.sum(self.layer3_weights_gradient,axis=0)*(1/batch_size)
        
        self.layer3_bias-=n*cp.sum(self.layer3_bias_gradient,axis=0)*(1/batch_size)
      
    def train(self,images,labels,num_images,s,batch_size):
        
            
   
        for i in range(3):
      
           
            self.run(images,0.5,True)
          
            

            layer3_reshaped=self.layer3.reshape(self.layer3.shape[0],self.layer3.shape[1])
            self.loss+=cp.sum((labels-layer3_reshaped)**2)*(1/num_images) 
            
            self.findgradient(labels,True,num_images,s)
            
            
            s=time.time()
            self.gradient_descent(0.1767766,num_images,s,batch_size)
            #self.reset()
            # self.layer1_weights_gradient=np.zeros(self.layer1_weights.shape)
            # self.layer1_bias_gradient=np.zeros(self.layer1_bias.shape)
            # self.layer2_weights_gradient=np.zeros(self.layer2_weights.shape)
            # self.layer2_bias_gradient=np.zeros(self.layer2_bias.shape)
            # self.layer3_weights_gradient=np.zeros(self.layer3_weights.shape)
            # self.layer3_bias_gradient=np.zeros(self.layer3_bias.shape)
   
        return
    def reset(self):
        self.layer1=self.layer2=self.layer3=self.flatten=None
        
        return
    def checkaccuracy(self,x_test,y_test,num_images,s):
       
        
        tot=x_test[0].shape[0]
        cor=0
        # s/=6
        iter=len(x_test)
       
        for i in range(iter):
            guess=self.run(cp.array(x_test[i]),0,False)
            
            # self.loss_test+=((np.sum((self.layer1_weights)**2)+np.sum((self.layer2_weights)**2)+np.sum((self.layer3_weights)**2)) *(s/(2*num_images)) )
            ans=cp.array(y_test[i])

            layer3_reshaped=self.layer3.reshape(self.layer3.shape[0],self.layer3.shape[1])
            self.loss_test+=cp.sum((ans-layer3_reshaped)**2)*(1/num_images) 
          
            
               
            n_values=10
            one_hot_encoded_guess= cp.eye(n_values)[guess]
        
            one_hot_encoded_guess[cp.arange(guess.size),guess] = 1
        
            cor+=cp.sum(cp.max(ans-one_hot_encoded_guess,axis=1))
        
        return (1-cor/num_images)*100
    def getparameters(self):
        return {0:self.layer1_weights.tolist(),1:self.layer1_bias.tolist(),2:self.layer2_weights.tolist(),3:self.layer2_bias.tolist(),4:self.layer3_weights.tolist(),5:self.layer3_bias.tolist()}
    def resetloss(self):
        self.loss=0
    def resetloss_test(self):
        self.loss_test=0

def update_file(data,acc):
    
    with open('data.json','w') as f:
        
        json.dump(data,f)
        f.close()
    print('New parameters saved. Best accuracy: '+ str(acc))
    print("\n")
   

isTrained=True
epoch=400
batch_size=125
num_images=60000
num_test_images=10000
lambda_val=6
num_batch_iteratons=num_images//batch_size 

(x_train,y),(x_test,y2)=tf.keras.datasets.fashion_mnist.load_data()

x_train=x_train/255
x_test=x_test/255
x_train=np.array_split(x_train,60000//batch_size)
y_train = np.zeros((y.size, y.max()+1))
y_train[np.arange(y.size),y] = 1
y_train=np.array_split(y_train,60000//batch_size)
y_test = np.zeros((y2.size, y2.max()+1))
y_test[np.arange(y2.size),y2] = 1
x_test=np.array_split(x_test,10000//100)

y_test=np.array_split(y_test,10000//100)


mempool.free_all_blocks()
pinned_mempool.free_all_blocks()

print('Images Loaded')
if isTrained:
    
    with open('data.json','r') as f:
        data=json.load(f)
    myNetwork=(NueralNetwork([],isTrained))
else:
    myNetwork=NueralNetwork([784,256,128,10],isTrained)




begin_accuracy=myNetwork.checkaccuracy(x_test,y_test,num_test_images,lambda_val)
print('Initial test loss: '+str(myNetwork.loss_test))
myNetwork.resetloss_test()
best_accuracy=begin_accuracy


print('Initial Accuracy: '+ str(begin_accuracy))
print("\n")

for x in range(epoch):
  
    s=time.time()
    curr_img=0
    for i in range(num_batch_iteratons):
        
      
        myNetwork.train(cp.array(x_train[i]),cp.array(y_train[i]),num_images,lambda_val,batch_size)
    #myNetwork.reset()   
    print('Epoch '+ str(x+1)+' Done | Time Taken: '+ str(time.time()-s)+"s")
  
    acc=myNetwork.checkaccuracy(x_test,y_test,num_test_images,lambda_val)
    print("Loss of training data (L2 loss omitted): "+ str(myNetwork.loss) + " | Loss of test data (L2 loss omitted): "+ str(myNetwork.loss_test))
    myNetwork.resetloss()    
    myNetwork.resetloss_test()
    if best_accuracy<acc:
        best_accuracy=acc
        params=myNetwork.getparameters()
      
        update_file(params,acc)
    else:
        print(" Accuracy: " + str(acc))
        print("\n")
       
