import numpy as np
from numpy.random import binomial
import math

def log(x):
	return math.log(x, 2)


#calculates entropy for given bernoullie distribution of rate p
def calculate_entropy(p):
	return -(p*log(p)+ (1-p)*log(1-p))


#visualization of entropy of a bernoullie random variable 
def Entropy_visual(maxVal):
    x=[]
    I = []
    for i in range(1, maxVal):
        I.append(i/maxVal)
        x.append(calculate_entropy(i/maxVal))
    return x, I

# Generate a sequence of binomial strings given an input length. Return type list 
def generate_sequence(Len, p):
	return binomial(n=1, p=p,size=Len)

def sample_mean_log_prob(sequence_vector, p):
	sum = 0
	for x in sequence_vector:
		sum += (log(p) if x==1 else log(1-p))
	return -sum/len(sequence_vector)

def generate_examples(num_ex, Len, p):
	X_ = []
	for i in range(num_ex):
		X_.append(generate_sequence(Len, p))
	return X_


def get_typical_set(X_train, epsilon, p):
	entropy = calculate_entropy(p)
	Typical_set = []
	else_set= []
	for x_train in X_train:
		if math.abs(sample_mean_log_prob(x_train, p)-entropy)<= epsilon:
			Typical_set.append(x_train)
		else:
			else_set.append(x_train)
	return Typical_set, else_set
	