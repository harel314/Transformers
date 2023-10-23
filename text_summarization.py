#%%
import pandas as pd
import textwrap
from transformers import pipeline

def wrap(x):
    return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)
def print_summary(doc):
    result = summarizer(doc.iloc[0])
    print(wrap(result[0]['summary_text']))
#%%
df_ = pd.read_csv('newsArticles/bbc_news_data.csv',sep="\t")
df = df_[['category','content']].copy()
#%%
doc = df[df['category']=='entertainment']['content'].sample(random_state=42)
print(wrap(doc.iloc[0]))
# %%
summarizer = pipeline("summarization")
#%%
print_summary(doc)
