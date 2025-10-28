import numpy as np
import pickle
import pandas as pd

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

def initialize_weights(input,output):
    limit = np.sqrt(6 / (input + output))

    W = np.random.uniform(-limit,+limit,(input,output))
    b = np.zeros((1, output))
    return W,b



def softmax(z):
    z= z - np.max(z,axis=1,keepdims=True)
    expz = np.exp(z)
    return expz/np.sum(expz,axis=1,keepdims=True)

def forward_propagation(X,W,b):
    z = np.dot(X,W) + b
    return softmax(z)

def background_propagation(X,W,b,lr,y_predict,y_true):
    batch_size = X.shape[0]
    W = W + lr*np.dot(X.T ,(y_true-y_predict)) / batch_size
    b = b + lr * np.sum(y_true - y_predict, axis=0, keepdims=True) / batch_size
    return W,b

def train(X,Y,lr,epochs,batch_size):
    W,b = initialize_weights(X_train.shape[1],Y_train.shape[1])
    for epoch in range(epochs):

        idx = np.random.permutation(X.shape[0])
        X, Y = X[idx], Y[idx]

        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            Y_batch = Y[start:end]

            y_pred = forward_propagation(X_batch, W, b)
            W, b = background_propagation(X_batch, W, b, lr, y_pred, Y_batch)

        if epoch % 20 == 0:
            lr = lr * 0.98
        y_predict = forward_propagation(X,W,b)
        W,b = background_propagation(X,W,b,lr,y_predict,Y_train)

    return W,b

W,b = train(X_train,Y_train,0.2,150, 128)

def calculate(X,W,b):
    prediction = forward_propagation(X,W,b)
    return np.argmax(prediction,axis=1)
res = calculate(X_test,W,b)

df = pd.DataFrame({"ID": np.arange(len(res)), "target": res})
df.to_csv("submission.csv", index=False)