#%% imports
from transformers import pipeline
import textwrap
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
#functions
def wrap(x):
    return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)

#%% set the dataframe
df_ = pd.read_csv('newsArticles/bbc_news_data.csv',sep="\t")
df = df_[['category','content']].copy()
df.head(1)
#%% extact the labels
labels = set(df['category'])
# choose specific category
label = 'business'
texts = df[df['category']==label]['content']
ix = np.random.choice(texts.shape[0])
doc = texts.iloc[ix]
pprint(wrap(doc))

# %% upload the model
mlm = pipeline('fill-mask')
mlm('Someone please put an end to <mask>')
#%% txt example taken from doc:

text = "The UK's biggest brewer, Scottish and Newcastle (S&N), is to buy " + \
 "37.5% of India's United <mask> in a deal worth 4.66bn rupees " \
 '($106m:£54.6m).'
mlm(text)

#%% ========== Automatic masking  of words in a doc ==========

#get frequencies of the letters
doc_letters = doc.split(' ')
all_freq = {}
for i in doc_letters:
 if i in all_freq:
  all_freq[i] += 1
 else:
  all_freq[i] = 1
res = min(all_freq, key = all_freq.get)
plt.bar(all_freq.keys(), all_freq.values(), 1.0, color='g')
# replace statistically all words that shows twice with mask
for key,value in all_freq.items():
    if value == 2:
      print(key)
      doc = doc.replace(key,"<mask>")

doc
#%%
