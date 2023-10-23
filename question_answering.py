#%%
from transformers import pipeline

qs = pipeline("question-answering")
#%%
ctx = "Today I don't feel like doing anything"
q = "what do I feel today?"

qs(context = ctx,question=q)
