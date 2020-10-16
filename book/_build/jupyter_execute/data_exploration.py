#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df_orl = pd.read_csv("data/parameter_igt_orl.csv")
df_orl["subjID_label"] = df_orl["subjID"].astype("category").cat.codes # number label for old/young
df_orl.head()


# # Paramters
# - A+: Reward Learning Rate
# - A-: Punishment Learning Rate
# - K: Decay
# - BetaF: Outcome freq
# - BetaP: Outcome perseverance
# 

# In[3]:


pd.plotting.scatter_matrix(df_orl[["A+", "A-", "K", "BetaF", "BetaP"]], figsize=(15,15), hist_kwds=dict(bins=50), c=df_orl["subjID_label"], cmap="Set1")
plt.show()


# In[ ]:




