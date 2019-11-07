import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import pickle

import time
from tqdm import tqdm
from data_loader import fetch_data
import vsmlib

unk = '<UNK>'

path_to_vsm = "data/word_linear_glove_500d"
vsm = vsmlib.model.load_from_dir(path_to_vsm)

class RNN(nn.Module):
	def __init__(self, h1, h2, output_size, input_size,  layers): # Add relevant parameters
		super(RNN, self).__init__()
		# Fill in relevant parameters
		self.h1 = h1
		self.h2 = h2
		self.layers = layers
		self.output_size = output_size

		self.rnn = nn.RNN(input_size, self.h1, num_layers=self.layers, bidirectional=False)
		self.activation = nn.ReLU()
		self.full1 = nn.Linear(self.h1, self.h2)
		self.full2 = nn.Linear(self.h2, self.output_size)
		# Ensure parameters are initialized to small values, see PyTorch documentation for guidance
		self.softmax = nn.LogSoftmax()
		self.loss = nn.NLLLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def forward(self, inputs): 
		#begin code
		self.batch_size = inputs.size(1)
		hidden = torch.zeros(self.layers, self.batch_size, self.h1)
		z1, hidden = self.rnn(inputs,hidden)
		a1 = z1[0][-1]
		z2 = self.full1(hidden)
		# a2 = self.activation(z2)
		# z3 = self.full2(a2)
		predicted_vector = self.softmax(z2) # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
		#end code
		return predicted_vector

# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)

def getEncoding(word): 
    if(vsm.has_word(word)):
        return vsm.get_row(word)
    else:
        return np.zeros(500)   

def preprocessData(train_data):
	data = []
	for document, y in train_data:
		seq = []
		for word in document:
			seq.append(getEncoding(word))
		data.append((torch.from_numpy(np.array([seq])),y))
	return data 

def main(h1, h2, number_of_epochs): # Add relevant parameters
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}	
	train_data = preprocessData(train_data)
	valid_data = preprocessData(valid_data)

	# Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
	# Further, think about where the vectors will come from. There are 3 reasonable choices:
	# 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
	# 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
	# 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
	# Option 3 will be the most time consuming, so we do not recommend starting with this

	model = RNN(h1, h2, 5, train_data[0][0].size()[2], 1) # Fill in parameters
	optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.0) 
	minibatch_size = 2

	for epoch in range(number_of_epochs): # How will you decide to stop training and why
		model.train()
		# You will need further code to operationalize training, ffnn.py may be helpful
		correct = 0
		total = 0
		start_time = time.time()
		print("Training started for epoch {}".format(epoch + 1))

		random.shuffle(train_data) # Good practice to shuffle order of training data
		N = len(train_data) 
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(minibatch_size):
				input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
				# print(input_vector.shape)
				predicted_vector = model(input_vector.float())
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
				# example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
				# if loss is None:
				# 	loss = example_loss
				# else:
				# 	loss += example_loss
				# loss = loss / minibatch_size	
				loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
				loss.backward()
				optimizer.step()
		print("Training avg loss {}".format(loss))
		print("Training completed for epoch {}".format(epoch + 1))
		print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Training time for this epoch: {}".format(time.time() - start_time))

		start_time = time.time()
		correct = 0 
		total = 0
		# You will need to validate your model. All results for Part 3 should be reported on the validation set. 
		for i in tqdm(range(len(valid_data))):
			input_vector, gold_label = valid_data[i]
			predicted_vector = model(input_vector.float())
			predicted_label = torch.argmax(predicted_vector)
			correct += int(predicted_label == gold_label)
			total += 1
			example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
			if loss is None:
				loss = example_loss
			else:
				loss += example_loss
		loss = loss / len(valid_data)
		print("Validation avg loss {}".format(loss))
		print("Validation completed for epoch {}".format(epoch + 1))
		print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Validation time for this epoch: {}".format(time.time() - start_time))