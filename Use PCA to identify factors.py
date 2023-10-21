#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from openbb_terminal.sdk import openbb
from sklearn.decomposition import PCA


# In[2]:


plt.style.use("default")
plt.rcParams["figure.figsize"] = [5.5, 4.0]
plt.rcParams["figure.dpi"] = 140
plt.rcParams["lines.linewidth"] = 0.75
plt.rcParams["font.size"] = 8


# In[3]:


#gold: NEM, RGLD, SSRM, CDE
# healthcare: LLY UNH, JNJ, MRK
symbols = ["NEM", "RGLD", "SSRM", "CDE", "LLY", "UNH", "JNJ", "MRK"]
data = openbb.economy.index(
    symbols, 
    start_date="2020-01-01", 
    end_date = "2022-12-31"
)


# In[4]:


returns = data.pct_change().dropna()


# In[5]:


pca = PCA(n_components=3)
pca.fit(returns)


# In[7]:


pct = pca.explained_variance_ratio_
pca_components = pca.components_


# In[10]:


cum_pct = np.cumsum(pct)
x = np.arange(1, len(pct) + 1, 1)

plt.subplot (1, 2, 1)
plt.bar(x, pct *100, align='center')
plt.title("contribution %")
plt.xticks(x)
plt.xlim([0, 4])
plt. ylim([0,100])

plt.subplot(1,2,2,)
plt.plot(x, cum_pct * 100, "ro-")
plt.title("Cumalative contribution (%)")
plt.xticks(x)
plt.xlim([0, 4])
plt.ylim([0, 100])


# #### From these principal components we can construct "statistical risk factors", similar to more conventional common risk factors. These should give us an idea of how much of the portfolio's returns comes from some unobservable statistical feature.

# In[12]:


X = np.asarray(returns)
factor_returns = X.dot(pca_components.T)
factor_returns = pd.DataFrame(
    columns= ["f1", "f2", "f3"],
    index = returns.index, 
    data=factor_returns
)

factor_returns


# In[14]:


factor_exposures = pd.DataFrame(
    index=["f1", "f2", "f3"], 
    columns=returns.columns, 
    data=pca_components
).T
factor_exposures


# In[16]:


factor_exposures.f1.sort_values().plot.bar()


# In[23]:


labels = factor_exposures.index
data = factor_exposures.values
plt.scatter(data[:, 0], data[:,1])
plt.xlabel("factor exposure of PC1")
plt.ylabel("factor exposure of PC2")

for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label, 
        xy=(x,y),
        xytext=(-20, 20), 
        textcoords= "offset points", 
        arrowprops=dict(
            arrowstyle= "->",
            connectionstyle="arc3, rad=0"
        )
    )


# In[ ]:




