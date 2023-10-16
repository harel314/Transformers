#%%
from transformers import pipeline
import pandas as pd
import torch
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
# some functions:
def categorize_rating(rating):
    if rating > 2.5:
        return 'positive'
    else:
        return 'negative'

#%%
classifier = pipeline("sentiment-analysis")
type(classifier)
#%% pass a sequence
classifier("I don't want to work anymore today")

#%% pass a list
classifier([
        "This work is crazy",
        "few more tasks and this day is over"]
)
#%% ==============run on data with GPU ====================
device = 0 if torch.cuda.is_available() else "cpu"
classifier = pipeline("sentiment-analysis",device=device)
df_ = pd.read_csv('starbucksReviews/reviews_data.csv')
#%% get only the rating and reviews and show hist of the ratings
df_.head(1)
df = df_[['Rating','Review']].copy()
df.hist()
#%% sort the dataframe with sentiment and targets
df['sentiment'] = df['Rating'].apply(categorize_rating) 
target_map = {'positive':1,'negative':0}
df['target'] = df['sentiment'].map(target_map)
len(df)
#%% get predictions
texts = df['Review'].tolist()
predictions = classifier(texts)
predictions[0] # show an example
#%% evaluate the predictions
# transform scores to probabilites
probs = [d['score'] if d['label'].startswith('P') else 1-d['score']
         for d in predictions]
# get predictions from probs
preds = [1 if d['label'].startswith('P') else 0 for d in predictions]
preds = np.array(preds)
# calc. accuracy
acc = np.mean(df['target']==preds)
print(f"{acc=}")
# draw confusion matrix
cm = confusion_matrix(df['target'],preds,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=['negative','positive'])
disp.plot()
# f1-score
f1s = f1_score(df['target'],preds)
f1s_inv = f1_score(1-df['target'],1-preds)
print(f"{f1s=},{f1s_inv=}")

# roc
roc_auc_s = roc_auc_score(df['target'],probs)
print(f"{roc_auc_s=}")
