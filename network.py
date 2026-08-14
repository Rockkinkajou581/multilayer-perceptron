import numpy as np
from activations import ReLu, ReLu_grad, softmax
from losses import categorical_cross_entropy

class NeuralNetwork:
    #layer sizes of the form [720, 350, ... ] representing input dimensions and output dimensions for each layer
    def __init__(self, layer_sizes, batch_size):
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.weights = []
        self.bias = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.normal(loc= 0.0, scale=np.sqrt(2 / layer_sizes[i]), size = (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.bias.append(b)
        
        self.activation = []
        self.preactivation = []

        self.grad = []
        self.grad_W = []
        self.grad_b = []

    def feed_forward(self, A_zero):
        self.activation = [A_zero]
        self.preactivation = []

        A = A_zero
        for i in range(len(self.weights)):
            Z_i = A @ self.weights[i] + self.bias[i]
            self.preactivation.append(Z_i)
            if(i == len(self.weights) - 1):
                A = softmax(Z_i)
            else:
                A = ReLu(Z_i)
            self.activation.append(A)
    
    def back_prop(self, A_zero, Y_true):
        self.grad_b = []
        self.grad_W = []

        self.feed_forward(A_zero)
        grad_base = (1 / self.batch_size) * (self.activation[-1] - Y_true)
        
        L = len(self.weights)

        self.grad = [None] * L
        self.grad[L - 1] = grad_base


        for i in range(L - 2, -1, -1):
            self.grad[i] = (self.grad[i+1] @ self.weights[i+1].T) * ReLu_grad(self.preactivation[i])

        for i in range(L):
            grad_W_i = np.transpose(self.activation[i]) @ self.grad[i] 
            grad_b_i = np.sum(self.grad[i], axis= 0, keepdims=True)

            self.grad_b.append(grad_b_i)
            self.grad_W.append(grad_W_i)
    def gradient_descent(self, A_zero, Y_true):
        lr = 0.05
        self.back_prop(A_zero, Y_true)
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.grad_W[i]
            self.bias[i] -= lr * self.grad_b[i]

    def validate(self, X_validate, Y_validate):

        self.feed_forward(X_validate)
        return categorical_cross_entropy(self.activation[-1], Y_validate)

    def evaluate_epoch(self, Y_true, epoch, loss):
        predictions = np.argmax(self.activation[-1], axis=1)   # predicted class per example
        true_labels = np.argmax(Y_true, axis=1)                # actual class per example
        accuracy = np.mean(predictions == true_labels)

        print(f"{epoch:<8}{loss:<10.4f}{accuracy:<10.2%}")
    

    def count_dead_neurons(self, X):
        self.feed_forward(X)
        dead_counts = []
        for Z in self.preactivation[:-1]:
            Z = Z <= 0
            percent = np.mean(Z, axis= 0)
            dead_counts.append(int(np.sum(percent >= 0.90)))
        return dead_counts


            


    def evaluate_final(self, X, Y_true):
        self.feed_forward(X)
        print(f"{'':<10}{'Precision':<12}{'Recall':<10}{'Fscore':<8}")
        predictions = np.argmax(self.activation[-1], axis=1)   # predicted class per example
        true_labels = np.argmax(Y_true, axis=1)   

        for c in range(self.layer_sizes[-1]):

            indices = np.where(predictions == c)[0]
            precision = np.mean(true_labels[indices] == c)
           
            indices = np.where(true_labels == c)[0]
            recall = np.mean(predictions[indices] == c)
            

            f_score = (2 * recall * precision) / (recall + precision)
            
            print(f"{'Class ' + str(c):<10}{precision:<12.2%}{recall:<10.2%}{f_score:<8.2%}")
       