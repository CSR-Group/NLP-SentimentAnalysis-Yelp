import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import pickle
import time
import vsmlib

from data_loader import fetch_data
from util import *
from torch.nn import init
from tqdm import tqdm
from torch.autograd import Variable
import  matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.input_dim = 500
        self.hidden_dim = 5
        self.num_rnn_layers = 1
        self.nonlinearity = 'tanh'
        self.log_softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()
        self.rnn = nn.RNN(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_rnn_layers, batch_first=True, nonlinearity=self.nonlinearity)
        
    def get_loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
    
    def forward(self, inputs):
        h0 = Variable(torch.zeros(self.num_rnn_layers, inputs.size(0), self.hidden_dim))
        out, hn = self.rnn(inputs, h0)
        return self.log_softmax(out[:, -1, :])


def performTrain(model, optimizer, train_data):
    random.shuffle(train_data)
    loss = None
    correct = 0
    optimizer.zero_grad()

    for i in tqdm(range(len(train_data))):
        input_vector, gold_label = train_data[i]
        predicted_vector = model(input_vector.float())
        predicted_label = torch.argmax(predicted_vector)
        correct += int(predicted_label == gold_label)
        instance_loss = model.get_loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
        if(loss is None):
            loss = instance_loss
        else:
            loss += instance_loss
    loss = loss / len(train_data)
    loss.backward()
    optimizer.step()
    accuracy = (correct / len(train_data)) * 100
    return loss.data, accuracy

def validate(model, val_data):
    correct = 0
    loss = None
    for i in tqdm(range(len(val_data))):
        input_vector, gold_label = val_data[i]
        predicted_vector = model(input_vector.float())
        predicted_label = torch.argmax(predicted_vector)
        correct += int(predicted_label == gold_label)
        instance_loss = model.get_loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
        if(loss is None):
            loss = instance_loss
        else:
            loss += instance_loss
    loss = loss / len(val_data)
    accuracy = (correct / len(val_data)) * 100
    return loss.data, accuracy

def main(num_epoch = 15):
    train_data,val_data = getMinTrainingAndValData(10)
    model = RNN()
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epoch):
        model.train()
        train_loss, train_accuracy = performTrain(model, optimizer, train_data)
        val_loss, val_accuracy = validate(model, val_data)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy))
        print("Training validation for epoch {}: {}".format(epoch + 1, val_accuracy))

    # visualization loss 
    iteration_list = [i+1 for i in range(num_epoch)]
    plt.plot(iteration_list,train_loss_history)
    plt.xlabel("Number of iteration")
    plt.ylabel("Training Loss")
    plt.title("RNN: Loss vs Number of iteration")
    plt.show()

    # visualization accuracy 
    plt.plot(iteration_list,val_loss_history,color = "red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Validation loss")
    plt.title("RNN: Loss vs Number of iteration")
    plt.savefig('graph.png')
    plt.show()


main()