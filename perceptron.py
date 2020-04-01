import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.out=[0]
        self.synaptic_weights = 2 * np.random.random((3,1)) -1

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_deri(self,x):
        return x*(1-x)

    def think(self,input):
        input=input.astype(float)
        output =self.sigmoid(np.dot(input,self.synaptic_weights))
        return output

    def train(self,training_input,training_output,training_iterations):
        for i in range(training_iterations):
            output =self.think(training_input)
            error =training_output-output
            adjustments =np.dot(training_input.T,error*self.sigmoid_deri(output))
            self.synaptic_weights+=adjustments
            self.out=np.append(self.out,(1-self.think(np.array([1,1,1])))*100)

#Start of Main Function()

if __name__ == "__main__":

   neural_network = NeuralNetwork()
   print("Random synaptic weights")
   print(neural_network.synaptic_weights)

   training_input =np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
   training_output =np.array([[0,1,1,0]]).T

   neural_network.train(training_input,training_output,6000)
   print("synaptic weights after training")
   print(neural_network.synaptic_weights)
   print(neural_network.out)
   x=np.arange(0,6001)

   plt.plot(x,neural_network.out)
   plt.title('Error Rate')
   plt.ylabel('Error ->')
   plt.xlabel('Iterations ->')
   plt.show()
   A=str(input("Input 1: "))
   B=str(input("Input 2: "))
   C=str(input("Input 3: "))

   print("New situation: input data =",A,B,C)
   print("Output data")
   print(np.around(neural_network.think(np.array([A,B,C]))))
