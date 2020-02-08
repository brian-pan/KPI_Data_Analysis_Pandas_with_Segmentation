#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Load data
# Name the data frame as df for simplicity
df = pd.read_csv("bank-full-add-data.csv", sep = ';')


# In[4]:


# Next do the exploratory data ananysis (EDA)

# Have a look at data
df.head()


# In[5]:


df.shape


# In[7]:


# Find out any missing columns
df.isnull().sum()


# In[9]:


# Our goal is modeling "y"
# Check variable "y"
df["y"].value_counts()


# In[14]:


# Find the conversion rate
# Create a column using 0 (no) and 1 (yes)
df["converCol"] = df["y"].apply(lambda x: 1 if x == "yes" else 0)


# In[17]:


df[["y", "converCol"]]


# In[18]:


df.head(10)


# In[20]:


# Once we made the column of 0 and 1, 
# we can sum to get a total number of yes
numeraConvRate = df["converCol"].sum()


# In[22]:


print(numeraConvRate)


# In[24]:


denomConvRate = df.shape[0]


# In[25]:


print(denomConvRate)


# In[27]:


conversionRate = numeraConvRate / denomConvRate * 100
print(conversionRate)


# In[28]:


# Analyze how these conversion rates vary among different age groups.
df.groupby(["age"])["converCol"].sum()


# In[29]:


df.groupby(["age"])["converCol"].count()


# In[30]:


converRatebyAge = df.groupby(["age"])["converCol"].sum()/df.groupby(["age"])["converCol"].count()


# In[31]:


print(converRatebyAge)


# In[45]:


# Plots
ax = converRatebyAge.plot(figsize = (11, 8), grid = True,
                         title = "Conversion Rate by ages",
)

ax.set_xlabel("age (years old)")
ax.set_ylabel('Conversion Rate (%)')
plt.show()


# In[49]:


df.groupby('age')['converCol'].count()


# In[72]:


df["groupedAge"] = df['age'].apply(
    lambda x: '[17, 30)' if x < 30 else '[30, 40)' if x < 40 \
        else '[40, 50)' if x < 50 else '[50, 60)' if x < 60 \
        else '[60, 70)' if x < 70 else '70+'
)


# In[73]:


AgeInterval = df.groupby(["groupedAge"]
          )["converCol"].sum() / df.groupby(
    ["groupedAge"])["converCol"].count() * 100


# In[74]:


print(AgeInterval)


# In[75]:


ax = AgeInterval.loc[
    ['[17, 30)', '[30, 40)', '[40, 50)', '[50, 60)', '[60, 70)', '70+']
].plot(
    kind='bar',
    color='darkblue',
    grid=True,
    figsize=(11, 8),
    title='Conversion Rates by Age Intervals'
)

ax.set_xlabel('age Intervals (years old)')
ax.set_ylabel('New Conversion Rate (%)')

plt.show()

