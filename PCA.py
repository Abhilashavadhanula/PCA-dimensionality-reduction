#!/usr/bin/env python
# coding: utf-8

# ## Principal Component Analysis
#  Since this isn't exactly a full machine learning algorithm, but instead an unsupervised learning algorithm.
# but no full machine learning project (although we will walk through the cancer set with PCA).

# In[1]:


##importing Librarys 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# Let's work with the cancer data set again since it had so many features.

# In[2]:


from sklearn.datasets import load_breast_cancer


# In[5]:


cancer=load_breast_cancer()


# In[ ]:


## it is in the form of dict


# In[7]:


cancer.keys()


# In[9]:


print(cancer['DESCR'])


# In[10]:


## converting it into dataframe
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])


# In[12]:


df.head()


# ### PCA Visualization
# As we've noticed before it is difficult to visualize high dimensional data, we can use PCA to find the first
# two principal components, and visualize the data in this new, two-dimensional space, with a single scatter-plot. 
# Before we do this though, we'll need to scale our data so that each feature has a single unit variance.

# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


scalar=StandardScaler()
scalar.fit(df)


# In[17]:


scaled_data=scalar.transform(df)


# PCA with Scikit Learn uses a very similar process to other preprocessing functions that come 
# with SciKit Learn. We instantiate a PCA object, find the principal components using the fit method, then apply the rotation and dimensionality reduction by calling transform().
# 
# We can also specify how many components we want to keep when creating the PCA object.

# ### now we apply PCA

# In[18]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)


# In[19]:


pca


# In[20]:


pca.fit(scaled_data)


# - now we can transform the data into 2 components

# In[21]:


x_pca = pca.transform(scaled_data)


# In[22]:



scaled_data.shape


# In[23]:


x_pca.shape


# We've reduced 30 dimensions to just 2! Let's plot these two dimensions out!

# In[24]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


#  ### Interpreting the components
# Unfortunately, with this great power of dimensionality reduction, comes the cost of being able to easily understand what these components represent.
# 
# The components correspond to combinations of the original features, the components themselves are stored as an attribute of the fitted PCA object:

# In[25]:


pca.components_


# in this numpy matrix array, each row represents a principal component, and each column relates back to the original features. we can visualize this relationship with a heatmap:

# In[26]:


df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])


# In[27]:


plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)


# #### This heatmap and the color bar basically represent the correlation between the various feature and the principal component itself.
# 
# Conclusion
# Hopefully this information is useful to you when dealing with high dimensional data!

# In[ ]:




