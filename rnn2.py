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
        self.hidden_dim = 64
        self.output_dim = 5
        self.num_rnn_layers = 2
        self.nonlinearity = 'relu'
        self.log_softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()
        self.rnn = nn.RNN(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_rnn_layers, batch_first=True, nonlinearity=self.nonlinearity)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def get_loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
    
    def forward(self, inputs):
        h0 = Variable(torch.zeros(self.num_rnn_layers, inputs.size(0), self.hidden_dim))
        out, hn = self.rnn(inputs, h0)
        z1 = self.fc(hn[:, -1, :])
        return self.log_softmax(z1)


def performTrain(model, optimizer, train_data):
    random.shuffle(train_data)
    N = len(train_data)
    correct = 0
    total = 0
    totalloss = 0
    minibatch_size = 16

    for minibatch_index in tqdm(range(N // minibatch_size)):
        optimizer.zero_grad()
        loss = None
        for example_index in range(minibatch_size):
            input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
            predicted_vector = model(input_vector.float())
            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total +=1
            instance_loss = model.get_loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
            if(loss is None):
                loss = instance_loss
            else:
                loss += instance_loss
        loss = loss / minibatch_size
        loss.backward()
        optimizer.step()
        totalloss +=loss
    accuracy = (correct / total) * 100
    return totalloss/(N // minibatch_size), accuracy

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

def main(num_epoch = 10):
    beg_time = time.time()
    count = 0
    train_data,val_data = getTrainingAndValData()
    model = RNN()
    optimizer = optim.Adagrad(model.parameters(),lr=0.01)
    train_accuracy_history = []
    val_accuracy_history = []
    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epoch):

        # if os.path.exists("rnnmodel.pth"):
        #     state_dict = torch.load("model.pth")['state_dict']
        #     model.load_state_dict(state_dict)
        #     print("Successful")

        if len(train_loss_history)>1 and (train_loss_history[-1] < val_loss_history[-1]) and (train_loss_history[-1] < train_loss_history[-2]) and (val_loss_history[-1] > val_loss_history[-2]):
            break
        
        count += 1
        model.train()
        optimizer.zero_grad()
        start_time = time.time()
        train_loss, train_accuracy = performTrain(model, optimizer, train_data)
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        start_time = time.time()
        val_loss, val_accuracy = validate(model, val_data)
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_accuracy))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

		#saving model aftr every epoch
        # path = "rnnmodel.pth"
        # torch.save({'state_dict': model.state_dict()},path)
    
    print("Total time to Train")
    print(time.time()-beg_time)

    print(train_accuracy_history)
    print(val_accuracy_history)
    print(train_loss_history)
    print(val_loss_history)

    print("Number of Parameters")
    # Number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for p in model.parameters():
        if p.requires_grad:
            print(p.numel())
    print(pytorch_total_params)

    # training loss 
    iteration_list = [i+1 for i in range(count)]
    plt.plot(iteration_list,train_loss_history)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Loss")
    plt.title("RNN: Loss vs Number of Epochs")
    #plt.show()
    plt.savefig('train_loss_history.png')
    plt.clf()
    
    # training accuracy
    plt.plot(iteration_list,train_accuracy_history)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Accuracy")
    plt.title("RNN: Accuracy vs Number of Epochs")
    #plt.show()
    plt.savefig('train_accuracy_history.png')
    plt.clf()

    # validation loss 
    plt.plot(iteration_list,val_loss_history,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Loss")
    plt.title("RNN: Loss vs Number of Epochs")
    #plt.show()
    plt.savefig('val_loss_history.png')
    plt.clf()

    # training accuracy
    plt.plot(iteration_list,val_accuracy_history,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("RNN: Accuracy vs Number of Epochs")
    #plt.show()
    plt.savefig('val_accuracy_history.png')
    plt.clf()

main()