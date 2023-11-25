#%%
from transformers import pipeline
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import textwrap

from sklearn.metrics import roc_auc_score, f1_score,confusion_matrix
#%%
clf = pipeline('zero-shot-classification',device=0)
#%%
clf("I like hamburgers",candidate_labels=["carnivor","vegan","vegeterian"])
#%% hamburger taken from wikipedia
text = "A hamburger, or simply burger, is a food consisting of fillings—usually \
      a patty of ground meat, typically beef—placed inside a sliced bun or bread \
      roll. Hamburgers are often served with cheese, lettuce, tomato, onion, pickles,\
      bacon, or chilis; condiments such as ketchup, mustard, mayonnaise, relish, or a \
      'special sauce', often a variation of Thousand Island dressing; and are \
      frequently placed on sesame seed buns. A hamburger patty topped with cheese is \
      called a cheeseburger"
clf(text, candidate_labels=["food","sport","buisness"])
