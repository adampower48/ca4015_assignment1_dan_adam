#!/usr/bin/env python
# coding: utf-8

# # Data Exploration
# ## Data Prep

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[19]:


df_orl = pd.read_csv("data/parameter_igt_orl.csv")
df_orl["subjID_label"] = df_orl["subjID"].astype("category").cat.codes # number label for old/young
df_orl.head()


# ## Paramters
# - A+: Reward Learning Rate
# - A-: Punishment Learning Rate
# - K: Decay
# - $\beta_F$ (BetaF): Outcome freq
# - $\beta_P$ (BetaP): Outcome perseverance
# 

# ## Visualising the dataset

# ### Entire dataset

# Looking at the whole dataset, each parameter seems to follow a distribution nicely. A+ and A- are skewed to the left, K is more uniform but slightly skewed left, and both $\beta_F$ and $\beta_P$ roughly follow normal distributions.  
# There is also a clear difference in the variation between the groups. The young group is much more concentrated, and the old is much more spread out. 
# In this plot and all other plots in this book, the grey marks represent young people and the red points represent the old participants.
# 

# In[18]:


pd.plotting.scatter_matrix(df_orl[["A+", "A-", "K", "BetaF", "BetaP"]], figsize=(10,10), hist_kwds=dict(bins=50), c=df_orl["subjID_label"], cmap="Set1")
plt.show()


# The parameter distributions for the two groups are not the same. The A+ parameter for the old group is much more varied. The K parameter is skewed much more to the right for the old group, and the $\beta_P$ seems to actually have two distributions; One similar to the young group, and another concentrated around the value 10.

# ### Young Group

# In[16]:


df_orl_young = df_orl[df_orl["subjID"] == "young"]
pd.plotting.scatter_matrix(df_orl_young[["A+", "A-", "K", "BetaF", "BetaP"]], figsize=(10,10), hist_kwds=dict(bins=25), diagonal="hist")
plt.show()


# ### Old Group

# In[17]:


df_orl_old = df_orl[df_orl["subjID"] == "old"]
pd.plotting.scatter_matrix(df_orl_old[["A+", "A-", "K", "BetaF", "BetaP"]], figsize=(10,10), hist_kwds=dict(bins=25), diagonal="hist")
plt.show()


# ### Splitting up the old group

# We can split up the old group futher based on the two distributions we see for $\beta_P$. **$\beta_P$=7** is chosen here as the value to split them.  
# The other parameters do not seem to have any difference between these groups, with the exception of **K**, which is slightly skewed left or right depending on the group.
# We are dealing with less data as we drill down further, so it would be difficult to come to any solid conclusions by going deeper.

# In[22]:


beta_p_split = 7


# In[25]:


_df = df_orl_old[df_orl_old["BetaP"] <= beta_p_split]
pd.plotting.scatter_matrix(_df[ ["A+", "A-", "K", "BetaF", "BetaP"]], figsize=(10,10), hist_kwds=dict(bins=15), diagonal="hist")
plt.show()


# In[26]:


_df = df_orl_old[df_orl_old["BetaP"] > beta_p_split]
pd.plotting.scatter_matrix(_df[ ["A+", "A-", "K", "BetaF", "BetaP"]], figsize=(10,10), hist_kwds=dict(bins=15), diagonal="hist")
plt.show()

