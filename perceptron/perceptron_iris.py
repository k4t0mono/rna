#!/usr/bin/env python
# coding: utf-8

# # Perceptron Iris

# In[1]:


import csv
import random
from pprint import pprint
from prettytable import PrettyTable


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


full_dataset = []

# Import the dataset
with open('./iris.txt') as fl:
    reader = csv.reader(fl, delimiter=',')
    for row in reader:
        dt = [ float(x) for x in row[:-1] ]
        cl = row[-1]
        
        full_dataset.append(dt + [cl])

# choosing the two classes
c0 = 'Iris-setosa'
c1 = 'Iris-versicolor'

# Getting the subset
dataset = []
for dt in full_dataset:
    if dt[-1] == c0:
        dt[-1] = 0
    elif dt[-1] == c1:
        dt[-1] = 1
    else:
        continue
    
    dataset.append(dt)

# Make a 80% train and 20% test
test_size = int(len(dataset) * 0.2)
print('Test size: {}'.format(test_size))

# Shuffle for good measure
random.shuffle(dataset)

# Spliting test and train sets
test = dataset[:test_size]
test_x = [ x[:-1] for x in test ]
test_y = [ x[-1] for x in test ]

train = dataset[test_size:]
train_x = [ x[:-1] for x in train ]
train_y = [ x[-1] for x in train ]


# In[ ]:


p = Perceptron(4, lr=0.05)

p.fit(train_x, train_y)

print(p.weights)


# In[ ]:


# cmatrix[pred][real]
cmatrix = [
    [0, 0],
    [0, 0]
]

for i in range(test_size):
    pred = p.predict(test_x[i])
    cmatrix[pred][test_y[i]] += 1

x = PrettyTable()
x.field_names = [ '', 'real 0', 'real 1' ]
x.add_row([ 'infer 0', cmatrix[0][0], cmatrix[1][0] ])
x.add_row([ 'infer 1', cmatrix[0][1], cmatrix[1][1] ])

print(x)


# In[ ]:




