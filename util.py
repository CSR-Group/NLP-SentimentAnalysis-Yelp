import numpy as np
import torch
import torch.nn as nn
from data_loader import fetch_data
import vsmlib

path_to_vsm = "data/word_linear_glove_500d"
vsm = vsmlib.model.load_from_dir(path_to_vsm)

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

def getTrainingAndValData():
    train_data, valid_data = fetch_data()
    train_data = preprocessData(train_data)
    valid_data = preprocessData(valid_data)
    return train_data, valid_data

def getMinTrainingAndValData(percent):
    train_data, valid_data = fetch_data()
    train_data = train_data[0:len(train_data)//percent]
    train_data = preprocessData(train_data)
    valid_data = preprocessData(valid_data)
    return train_data, valid_data