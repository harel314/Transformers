#%%
from transformers import pipeline
from datasets import load_dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer

ner = pipeline("ner",aggregation_strategy='simple',device=0)
data = load_dataset("conll2003")
#%%
# %%
ner_feature = data["train"].features["ner_tags"]
label_names = ner_feature.feature.names

# %%
inputs = [words for words in data["train"]['tokens']]
targets = [words for words in data["train"]['ner_tags']]

# %%
detokenizer = TreebankWordDetokenizer()
ner(detokenizer.detokenize(inputs[0]))
