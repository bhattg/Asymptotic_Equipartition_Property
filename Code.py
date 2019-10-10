#!/usr/bin/env python
# coding: utf-8

# In[218]:


import numpy as np
from numpy.random import binomial
import math

def log(x):
	return math.log(x, 2)

def getValFromSeq(seq):
    intseq = [str(s) for s in seq]
    strVal = "".join(intseq)
    return int(strVal, 2)

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

def getTypicalPerc(TypSet, X_):
    a = len(set(TypSet))
    b = len(set(X_))
    return a/b, a, b

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
    Tot=[]
    for x_train in X_train:
        aa = getValFromSeq(x_train)
        Tot.append(aa)
        if abs(sample_mean_log_prob(x_train, p)-entropy)<= epsilon:
            Typical_set.append(aa)
    return Typical_set, Tot

    


# # Entropy Visualization for a bernoullie random variable 
# 
# #### Number of points chosen for visualization = 500

# In[241]:


x, I = Entropy_visual(500)
plt.plot(I,x)
plt.xlabel("p")
plt.ylabel("Entropy H")


# # Plotting Sample mean values for different 
# 
# It can be seen that as we increase the number variables in the sequence, then it moves closer and closer to the theoretical approximation values

# In[242]:


Len1 = 50
Len2 = 100
Len3 = 1000
Len4 = 5000

fig = plt.figure()
x_1=[]
for i in I:
    x_1.append(sample_mean_log_prob(generate_sequence(Len1, i),i))
x_2=[]
for i in I:
    x_2.append(sample_mean_log_prob(generate_sequence(Len2, i),i))
x_3=[]
for i in I:
    x_3.append(sample_mean_log_prob(generate_sequence(Len3, i),i))
x_4=[]
for i in I:
    x_4.append(sample_mean_log_prob(generate_sequence(Len4, i),i))


# In[243]:


plt.subplot(2, 2, 1)
plt.plot(I,x_1)
plt.subplot(2, 2, 2)
plt.plot(I,x_2)
plt.subplot(2, 2, 3)
plt.plot(I,x_3)
plt.subplot(2, 2, 4)
plt.plot(I,x_4)


# # Construction of Typical Set 
# ## Varying the value of epsilon and checking the size of typical set
# 
# * epsilon=0.1
# * Len =  100
# * num_ex=2000
# * p=0.2

# In[17]:


epsilon=0.1
Len =  [128, 256, 512, 1024]
num_ex=2048
p=0.2
Ty = []
El = []
for l in Len:
    X_ = generate_examples(num_ex=num_ex, p=p, Len=l)
    TypSet, elseSet  = get_typical_set(X_, epsilon=epsilon, p=p)
    Ty.append(TypSet)
    El.append(elseSet)
for i in range(4):
    print(len(Ty[i]))


# In[18]:


epsilon=0.05
Len =  [128, 256, 512, 1024]
num_ex=2048
p=0.2
Ty = []
El = []
for l in Len:
    X_ = generate_examples(num_ex=num_ex, p=p, Len=l)
    TypSet, elseSet  = get_typical_set(X_, epsilon=epsilon, p=p)
    Ty.append(TypSet)
    El.append(elseSet)
for i in range(4):
    print(len(Ty[i]))


# In[19]:


epsilon=0.01
Len =  [128, 256, 512, 1024]
num_ex=2048
p=0.2
Ty = []
El = []
for l in Len:
    X_ = generate_examples(num_ex=num_ex, p=p, Len=l)
    TypSet, elseSet  = get_typical_set(X_, epsilon=epsilon, p=p)
    Ty.append(TypSet)
    El.append(elseSet)
for i in range(4):
    print(len(Ty[i]))


# ## Varying the pmf "p" value and observing the size of Typical Set

# In[22]:


TT = []
EE = []
PP = [0.02*i for i in range(1, 50)]
for p in PP:
    epsilon=0.1
    Len = [128, 256, 512, 1024]
    num_ex=2000
    Ty = []
    El = []
    for l in Len:
        X_ = generate_examples(num_ex=num_ex, p=p, Len=l)
        TypSet, elseSet  = get_typical_set(X_, epsilon=epsilon, p=p)
        Ty.append(TypSet)
        El.append(elseSet)
    TT.append(Ty)
    EE.append(El)


# In[ ]:


epsilon_=[0.03,0.05, 0.08, 0.1, 0.15]
Len_ =  [8, 9, 10,11, 12]
num_ex=200000
p_=[0.1, 0.2, 0.25, 0.3]
TypicalSET = {}
X_TOTAL = {}
PERCENTAGE = {}
LENT = {}
LENX_ = {}
for Len in tqdm(Len_):
    for p in tqdm(p_):
        for epsilon in tqdm(epsilon_):
            X_ = generate_examples(num_ex=num_ex, p=p, Len=Len)
            TypSet, X_= get_typical_set(X_, epsilon=epsilon, p=p)
            percentage, lenT, lenX_= getTypicalPerc(TypSet,X_)
            key = (Len, p, epsilon)
            TypicalSET[key] = TypSet
            X_TOTAL[key] = X_
            PERCENTAGE[key] = percentage
            LENT[key]= lenT
            LENX_[key]= lenX_
            


# In[ ]:


for key in LENT.keys():
    print(str(key)+ " "+str(LENT[key]))


# In[191]:


k = (10, 0.3, 0.08)
plt.hist(x= TypicalSET[k], bins=1024)
plt.xlabel("Support of decimal representation of sequence")
plt.ylabel("Frequency")
plt.title("Frequency distribution")
plt.show()


# 
# ## Typical Set : Size and percentage analysis

# In[206]:


l = [8, 9, 10, 11, 12]
for a in l:
    k = (a, 0.2, 0.1)
    print("Size of typical set is {} with total being {} and percentage being {}".format(LENT[k], LENX_[k], 100*PERCENTAGE[k]))


# In[207]:


l = [8, 9, 10, 11, 12]
for a in l:
    k = (a, 0.1, 0.1)
    print("Size of typical set is {} with total being {} and percentage being {}".format(LENT[k], LENX_[k], 100*PERCENTAGE[k]))


# In[208]:


l = [8, 9, 10, 11, 12]
for a in l:
    k = (a, 0.25, 0.1)
    print("Size of typical set is {} with total being {} and percentage being {}".format(LENT[k], LENX_[k], 100*PERCENTAGE[k]))


# In[209]:


l = [8, 9, 10, 11, 12]
for a in l:
    k = (a, 0.3, 0.1)
    print("Size of typical set is {} with total being {} and percentage being {}".format(LENT[k], LENX_[k], 100*PERCENTAGE[k]))


# In[210]:


TS = [8, 9, 10, 11, 12]
TS1 = [17, 36, 45, 55, 272]
TS2 = [28, 36, 165, 165, 220]
TS3 = [84, 120, 120, 495, 715]


# In[215]:


plt.plot(Len_ , TS, label="0.1")
plt.plot(Len_ , TS1, label="0.2")
plt.plot(Len_ , TS2, label="0.25")
plt.plot(Len_ , TS3, label="0.3")
plt.xlabel("Length of Sequence")
plt.ylabel("Size of Typical Set")
plt.title("Variation of Size of Typical Set vs Length of the sequence")
plt.legend(loc='upper left')
plt.show()


# ## Typical Set size emperical estimation

# In[221]:


for n in Len_:
    for p in p_:
        print(str(n)+ " "+str(p)+" "+str(math.pow(2, n*calculate_entropy(p))))


# ### It can be clearly seen that these upper cap values that are determined by using the mathematical values of Entropy are far from the size that we saw. The reason is clear that the Length of Sequence 12 is not enough to have a proper approximation of sample mean to the Entropy value

# In[ ]:




