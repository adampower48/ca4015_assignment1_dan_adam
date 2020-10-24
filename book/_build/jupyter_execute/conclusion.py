#!/usr/bin/env python
# coding: utf-8

# # Conclusions

# ## Clustering
# In this paper, we investigate the clustering of parameters of just one of the models used in the Iowa Gambling task, the Outcome-Representation Learning model. Our reason of choice was based on which system had the most interesting parameters to examine and cluster. Our findings in the data exploration page further proved our hypothesis after plotting the distributions of the parameters against each other.
# 
# The most significant clustering result we discovered was when clustering BetaP and BetaK using K-Means. There were three distinct clusters of: 
# 1. A mainly old cluster of participants with a high perseverence.
# 2. Another mainly old cluster of people with a low perseverence.
# 3. A cluster of predominately young people centered around the middle with a more neutral BetaP score.
# 
# This followed the distribution of the two parameters where, people who had a lower perseverence and preferred to switch decks in the task generally had a higher win frequency. Although there isn't enough data to solidify this conclusion, this is the result we found significant.
# 
# There are many possibilities for future work for this model. First and foremost, a higher sample size will be needed to support any hypotheses gathered from the model and its clusters. More information on each of the participants is paramount in explaining the clusters of data. For example, some studies show that people with orbifrontal cortex dysfunction tend to have a high value for the perseverence metric. Further studies of the task and the participant information will result in the discovery of certain phenotypes' tendencies of actions during the task.
# 
# Exploring the paramater plots revealed their distributions. Based on the data given to us, they appear to follow: Normal distribution for $\beta_F$ and $\beta_P$, Beta($\alpha$,$\beta$) distribution for A+ and A-, and a Gamma distribution for the rate of decay K. More research into this will result in accurate data synthesis. There is a possibility that peoples' participation in the task will be predicted with high accuracy based on their information.

# In[ ]:




