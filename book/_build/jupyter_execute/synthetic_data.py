#!/usr/bin/env python
# coding: utf-8

# # Generating Synthetic Data

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


# In[2]:


df_orl = pd.read_csv("data/parameter_igt_orl.csv")
df_orl["subjID_label"] = df_orl["subjID"].astype("category").cat.codes # number label for old/young
df_orl.head()


# ## Fitting distributions to the parameters

# The [Beta(a,B) distribution](https://en.wikipedia.org/wiki/Beta_distribution) can be used to model a large variety of distributions.  
# It has shape parameters a and B, and location/scale parameters.  

# ### Modelling a single parameter

# In[3]:


a, b, loc, scale = beta.fit(df_orl["BetaP"])
a, b, loc, scale


# In[4]:


x = np.linspace(-15, 15, 100)
y = beta.pdf(x, a,b,loc=loc,scale=scale)

plt.hist(df_orl["BetaP"], density=True)
plt.plot(x, y)
plt.show()


# ### Applying to all parameters

# First we define a function to get the beta parameters

# In[5]:


def fit_beta(s: pd.Series):
    a, b, loc, scale = beta.fit(s)
    return pd.Series({
        "a": a,
        "b": b,
        "loc": loc,
        "scale": scale
        })

fit_beta(df_orl["BetaP"])


# Then we can apply this accross all of our parameters

# In[6]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', '\nbeta_params = df_orl[["A+", "A-", "K", "BetaF", "BetaP"]].apply(fit_beta)\nbeta_params')


# In[7]:


fig, ax = plt.subplots(2, 3, figsize=(15, 10))

for i, param in enumerate(beta_params.columns):
    row, col = divmod(i, 3)
    b_params = beta_params[param]

    # Generate points to draw beta
    x = np.linspace(df_orl[param].min(), df_orl[param].max(), 100)
    y = beta.pdf(x, b_params["a"], b_params["b"], loc=b_params["loc"], scale=b_params["scale"])

    # Draw plots
    ax[row, col].hist(df_orl[param], density=True, bins=25)
    ax[row, col].plot(x, y)
    
    # Formatting
    ax[row, col].set_title(param)
    
# Hide unused plot
ax[1, 2].set_visible(False)

plt.show()


# ### Generating new data

# By modelling these parameters, we can create new data by sampling from their distributions.  
# For now we assume that each parameter is independent, though this is probably not the case.  
# Here we generate 5000 fake data points.  

# In[8]:


synth_data = pd.DataFrame()
for param in beta_params.columns:
    b_params = beta_params[param]
    sample = beta.rvs(b_params["a"], b_params["b"], loc=b_params["loc"], scale=b_params["scale"], size=5000)
    synth_data[param] = sample
    
synth_data.head()


# In[9]:


pd.plotting.scatter_matrix(synth_data, figsize=(10,10), hist_kwds=dict(bins=50), alpha=0.2)
plt.show()


# ### Modelling the smaller groups

# As we saw when we explored the data, the young and old groups differ quite a bit, and the old group could be clustered into two sets.

# #### Young

# In[10]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', '\ndf_orl_young = df_orl[df_orl["subjID"] == "young"]\nbeta_params_young = df_orl_young[["A+", "A-", "K", "BetaF", "BetaP"]].apply(fit_beta)\nbeta_params_young')


# #### Old

# In[11]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', '\ndf_orl_old = df_orl[df_orl["subjID"] == "old"]\nbeta_params_old = df_orl_old[["A+", "A-", "K", "BetaF", "BetaP"]].apply(fit_beta)\nbeta_params_old')


# In[12]:


beta_p_split = 7


# Type A

# In[13]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', '\ndf_orl_old_a = df_orl[(df_orl["subjID"] == "old") & (df_orl["BetaP"] <= beta_p_split)]\nbeta_params_old_a = df_orl_old_a[["A+", "A-", "K", "BetaF", "BetaP"]].apply(fit_beta)\nbeta_params_old_a')


# Type B

# In[14]:


get_ipython().run_cell_magic('capture', '--no-stdout --no-display', '\ndf_orl_old_b = df_orl[(df_orl["subjID"] == "old") & (df_orl["BetaP"] > beta_p_split)]\nbeta_params_old_b = df_orl_old_b[["A+", "A-", "K", "BetaF", "BetaP"]].apply(fit_beta)\nbeta_params_old_b')


# #### Results

# In[15]:


# Function to draw row of plots
def draw_beta_fit(beta_params_df, data_df, ax_row, y_label, ):
    for i, param in enumerate(beta_params_df.columns):
        b_params = beta_params_df[param]
    
        # Generate points to draw beta
        x = np.linspace(data_df[param].min(), data_df[param].max(), 100)[1:-1]
        y = beta.pdf(x, b_params["a"], b_params["b"], loc=b_params["loc"], scale=b_params["scale"])

        # Draw plots
        ax_row[i].hist(data_df[param], density=True, bins=25)
        ax_row[i].plot(x, y)
        
    ax_row[0].set_ylabel(y_label)


# In[16]:


fig, ax = plt.subplots(4, 5, figsize=(25, 10))

# Plots
draw_beta_fit(beta_params_young, df_orl_young, ax[0], "Young")
draw_beta_fit(beta_params_old, df_orl_old, ax[1], "Old")
draw_beta_fit(beta_params_old_a, df_orl_old_a, ax[2], "Old - Type A")
draw_beta_fit(beta_params_old_b, df_orl_old_b, ax[3], "Old - Type B")

# Param titles
for i, param in enumerate(beta_params_young.columns):
    ax[0, i].set_title(param)
    
plt.show()


# In[ ]:




