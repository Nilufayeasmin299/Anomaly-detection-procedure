#!/usr/bin/env python
# coding: utf-8

# ### Step-1:

# In[1]:


# import library function
import numpy as np


# In[2]:


# Create the 'normal' array with random values between 0 and 10
normal = np.random.uniform(0, 10, size=(100, 1000))


# In[3]:


# Save the 'normal' array to a binary file named 'normal.bin'
with open('normal.bin', 'wb') as f:
    np.save(f, normal)


# In[4]:


# Create the 'abnormal' array with random values between 5 and 15
abnormal = np.random.uniform(5, 15, size=(100, 1000))


# In[5]:


# Save the 'abnormal' array to a binary file named 'abnormal.bin'
with open('abnormal.bin', 'wb') as f:
    np.save(f, abnormal)


# ### Step-2: 

# In[6]:


# Load the data from 'normal.bin' into the 'normal' array
with open('normal.bin', 'rb') as f:
    normal = np.load(f)


# In[7]:


# Load the data from 'abnormal.bin' into the 'abnormal' array
with open('abnormal.bin', 'rb') as f:
    abnormal = np.load(f)


# ### Step-3: 

# In[8]:


normal_length = len(normal)


# In[9]:


# Calculate the indices for splitting the data
train_index = int(0.9 * normal_length)
test_index = train_index + int(0.1 * normal_length)


# In[10]:


# Create the "training" array with 90% of the data from the "normal" array
training = normal[:train_index]


# In[11]:


# Create the "test" array with 10% of the data from the "normal" array and 10% from the "abnormal" array
test_normal = normal[train_index:test_index]
test_abnormal = abnormal[:test_index-train_index]


# In[12]:


#add 10% data from "normal" array and 10% data from "abnormal" array
test = np.concatenate((test_normal, test_abnormal), axis=0)


# ### Step-4:

# In[13]:


# import library function
from scipy.spatial.distance import euclidean


# In[14]:


# Calculate dissimilarity scores for each element in the "training" set
baseline = []
for i in range(len(training)):
    distances = []
    for j in range(len(training)):
        if i != j:
            distance = euclidean(training[i], training[j])
            distances.append(distance)
    distances.sort()
    dissimilarity_score = sum(distances[:5])
    baseline.append(dissimilarity_score)


# In[15]:


# Convert the "baseline" list to a NumPy array
baseline = np.array(baseline)


# ### Step-5 

# In[16]:


# Calculate dissimilarity scores and flag elements as normal or abnormal
predictions = []
for i in range(len(test)):
    distances = []
    for j in range(len(training)):
        distance = euclidean(test[i], training[j])
        distances.append(distance)
    distances.sort()
    dissimilarity_score = sum(distances[:5])
    
    if dissimilarity_score >= np.min(baseline) and dissimilarity_score <= np.max(baseline):
        predictions.append("normal")
    else:
        predictions.append("abnormal")


# In[17]:


# Print the algorithm's predictions
for i, prediction in enumerate(predictions):
    print(f"Element {i+1}: {prediction}")


# In[ ]:




