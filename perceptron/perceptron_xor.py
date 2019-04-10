#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Perceptron:
    def __init__(self, input_size, lr=0.95, epochs=100, bias=1):
        # Bias is the last entry
        self.weights = [0] * (input_size + 1)
        self.epochs = epochs
        self.lr = lr
        self.bias = bias
    
    # Binary activation function
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    # Makes weighted sum of the inputs
    def sum(self, x):
        if len(x) != len(self.weights):
            raise Exception(
                "Should have {} entries, got {}.".format(
                    len(self.weights) - 1, len(x) - 1
                )
            )
        s = 0
        for i in range(len(x)):
            s += x[i] * self.weights[i]
        
        return s
    
    # Predict the class for the inputs
    def predict(self, x):
        s = self.sum(list(x) + [self.bias])
        return self.activation(s)

    # Runs the training process
    def fit(self, x, y):
        for _ in range(self.epochs):
            for j in range(len(y)):
                x_ = x[j]
                y_ = self.predict(x_)
                e = y[j] - y_
                
                for k in range(len(self.weights) - 1):
                    self.weights[k] = self.weights[k] + self.lr * e * x_[k]
                self.weights[-1] = self.weights[-1] + self.lr * e * self.bias


# In[ ]:


# Perceptron OR, separa (0, 0) dos outros dados
p0 = Perceptron(2, lr=0.05)

entries_p0_x = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
]
entries_p0_y = (0, 1, 1, 1)

p0.fit(entries_p0_x, entries_p0_y)
print('p0.weights: {}'.format(p0.weights))


# In[ ]:


# Perceptron NAND, separa (1, 1) dos outros dados
p1 = Perceptron(2, lr=0.05)

entries_p1_x = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
]
entries_p1_y = (1, 1, 1, 0)

p1.fit(entries_p1_x, entries_p1_y)
print('p1.weights: {}'.format(p1.weights))


# In[ ]:


# Perceptron AND
p2 = Perceptron(2, lr=0.05)

entries_p2_x = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
]
entries_p2_y = (0, 0, 0, 1)

p2.fit(entries_p2_x, entries_p2_y)
print('p2.weights: {}'.format(p2.weights))


# In[ ]:


# Rede de perceptron multinível, as saídas de p0 e p1 são alimentas na entrada de p2
# a XOR b <=> (a OR b) AND (a NAND b)
for x in [ (0,0), (0,1), (1,0), (1,1) ]:
    print('x: {}'.format(x))
    
    a = p0.predict(x)
    b = p1.predict(x)
    
    r = p2.predict((a,b))
    print('r: {}'.format(r))
    print()

