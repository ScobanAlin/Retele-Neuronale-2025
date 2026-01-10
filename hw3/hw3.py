import pickle
import os
import pandas as pd
import numpy as np


train_file = "./extended_mnist_train.pkl"
test_file = "./extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)



train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)


test_data = []
for image, label in test:
    test_data.append(image.flatten())

X_train = np.array(train_data) / 255
y_train = np.array(train_labels)
Y_train = np.eye(10)[y_train]
X_test = np.array(test_data) / 255

def initialize_weights(input_dim=784, hidden_dim=100, output_dim=10):

    limit1 = np.sqrt(6 / (input_dim + hidden_dim))
    W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
    b1 = np.zeros((1, hidden_dim))

    limit2 = np.sqrt(6 / (hidden_dim + output_dim))
    W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
    b2 = np.zeros((1, output_dim))

    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(z):
    z= z - np.max(z,axis=1,keepdims=True)
    expz = np.exp(z)
    return expz/np.sum(expz,axis=1,keepdims=True)

def forward(X,W1,b1,W2,b2):
    Z1=np.dot(X,W1) +b1
    A1 = relu(Z1)
    Z2 = np.dot(A1,W2) + b2
    A2 = softmax(Z2)
    return A2 ,(Z1,A1,Z2,A2)

def backward(X, Y, variables_behind, W1, W2, l2_lambda=0.0005):

    Z1, A1, Z2, A2 = variables_behind
    m = X.shape[0]

    dZ2 = (A2 - Y) / m
    dW2 = np.dot(A1.T, dZ2) + l2_lambda * W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) + l2_lambda * W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


def train(X,Y,lr,epochs,batch_size):
    W1,b1,W2,b2= initialize_weights()
    for epoch in range(epochs):

        idx = np.random.permutation(X.shape[0])
        X, Y = X[idx], Y[idx]

        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            Y_batch = Y[start:end]

            y_pred, variables_behind  = forward (X_batch, W1, b1, W2, b2)

            dW1,db1 ,dW2,db2 = backward(X_batch, Y_batch, variables_behind,W1,W2)

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2



        if epoch % 15 == 0:
            lr = lr * 0.8

    return W1,b1,W2,b2

W1,b1,W2,b2 = train(X_train,Y_train,0.25,150, 128)

def calculate(X_test,W1,b1,W2,b2):
    prediction , _  = forward(X_test,W1,b1,W2,b2)
    return np.argmax(prediction,axis=1)
res = calculate(X_test,W1,b1,W2,b2)

df = pd.DataFrame({"ID": np.arange(len(res)), "target": res})
df.to_csv("submission.csv", index=False)