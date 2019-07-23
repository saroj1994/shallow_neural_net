import numpy as np
import matplotlib.pyplot as plt


image_size=28
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "C:\\Users\\lenovo\\Documents\\ds\\mnist\\"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",")






fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])




lr = np.arange(10)
for label in range(10):
    one_hot = (lr==label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)





lr = np.arange(no_of_different_labels)
# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)
# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99
################################################################
for i in range(10):
    img = train_imgs[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()



###########################################################



def sigmoid(z):
    return 1/(1+np.exp(-z))



class NeuralNetwork:
    def __init__(self, input_shape,output_shape,hidden_units,alpha):
        
        self.weights1   = np.random.randn(hidden_units,input_shape)*0.01 
        self.weights2   = np.random.randn(output_shape,hidden_units)*0.01                 

        
    def feedforward(self,inpu,y_shape,alpha):
        inpu      = np.array(inpu,ndmin=2).T
        
        output     = np.zeros(y_shape)
        
        layer1 = sigmoid(np.dot(self.weights1, inpu))
        output = sigmoid(np.dot(self.weights2,layer1))
        return output
    
    def train(self,inpu,y,alpha):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        inpu      = np.array(inpu,ndmin=2).T
        y          = np.array(y,ndmin=2).T
        output     = np.zeros(y.shape)
        
        
        
        layer1 = sigmoid(np.dot(self.weights1,inpu))
        output = sigmoid(np.dot(self.weights2,layer1))
        
        
        d_weights2 = alpha*np.dot(output-y,layer1.T)
        d_weights1 = alpha*np.dot(np.dot(self.weights2.T,output-y)*layer1*(1-layer1),inpu.T)

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
        
ann=NeuralNetwork(784,10,5,.01)        
for i in range(train_imgs.shape[0]):        
        ann.train(train_imgs[i],train_labels_one_hot[i],.1)
        
        