#%%
import numpy as np
from transformers import pipeline
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

#%%
data = load_dataset("gotchab/Spa-Eng-Finetuning")
#%%
d_train = data["train"]
# %%
eng2spa = {}
for ix,inputs in enumerate(d_train["inputs"]):
    if inputs not in eng2spa:
        eng2spa[inputs] = []
    eng2spa[inputs].append(d_train['targets'][ix])
    if ix == 1000:
        break
# %%
#%%
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize("como estas?".lower())
sentence_bleu([tokens],tokens)
#%%
smoother = SmoothingFunction()
sentence_bleu([tokens],tokens,smoothing_function=smoother.method4)
#%%
translator = pipeline("translation",model='Helsinki-NLP/opus-mt-en-es',device=0)
#%%
translator("It is so nice to hear from you!")
#%%
eng_phrases = list(eng2spa.keys())
translations = translator(eng_phrases)
#%%
translations[0]
#%%
eng2spa_tokens = {}
for eng,spa_list in eng2spa.items():
    spa_list_tokens = []
    for text in spa_list:
        tokens = tokenizer.tokenize(text.lower())
        spa_list_tokens.append(tokens)
    eng2spa_tokens[eng]=spa_list_tokens
scores = []

for eng,pred in zip(eng_phrases,translations):
    matches = eng2spa_tokens[eng]
    spa_pred = tokenizer.tokenize(pred["translation_text"].lower())
    score = sentence_bleu(matches,spa_pred)
    if score < 1e-50:
        score = 1
    scores.append(score)
#%%
plt.hist(scores,bins=50)

np.mean(scores)

#%%
def random_translation():
    ix = np.random.choice(len(eng_phrases))
    eng = eng_phrases[ix]
    print("EN: ",eng)

    translation = translations[ix]['translation_text']
    print("ES Translation:",translation)

    matches = eng2spa[eng]
    print("Matches: ", matches)

    print("BLEU: ",sentence_bleu(matches,translation))
#%%
random_translation()