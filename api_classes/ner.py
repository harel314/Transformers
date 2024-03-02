#%%
from transformers import pipeline
from datasets import load_dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
#%%
ner = pipeline("ner",aggregation_strategy='simple',device=0)
data = load_dataset("conll2003")

# %%
ner_feature = data["train"].features["ner_tags"]
label_names = ner_feature.feature.names

# %%
inputs = [words for words in data["train"]['tokens']]
targets = [words for words in data["train"]['ner_tags']]

# %%
detokenizer = TreebankWordDetokenizer()
ner(detokenizer.detokenize(inputs[0]))

# %%
from transformers import AutoTokenizer

checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# %%

begin2inside = {
    1:2,
    3:4,
    5:6,
    7:8,
}
def aligned_labels(labels,word_ids):
    aligned_labels = [];last_word = None
    for word in word_ids:
        if word is None:
            label = -100
        elif word != last_word:
            label = labels[word]
        else:
            label = labels[word]
        
        if label in begin2inside:
            label = begin2inside[label]

        aligned_labels.append(label)
        last_word = word
    return aligned_labels

def tokenize_fn(batch):
    tokenized_inputs = tokenizer(batch['tokens'], truncation=True,is_split_into_words=True)
    labels_batch = batch['ner_tags']
    aligned_labels_batch = []
    for i, labels in enumerate(labels_batch):
        word_ids = tokenized_inputs.word_ids(i)
        aligned_labels_batch.append(aligned_labels(labels,word_ids))
    tokenized_inputs['labels'] = aligned_labels_batch
    return tokenized_inputs
# %%
tokenized_datasets = data.map(
    tokenize_fn,
    batched=True,
    remove_columns=data["train"].column_names,
)
# %%
from transformers import DataCollatorForTokenClassification
from datasets import load_metric

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metric = load_metric("seqeval")

def compute_metrics(logits_and_labels):
    logits,labels = logits_and_labels
    preds = np.argmax(logits, axis=-1)

    str_labels = [[label_names[t] for t in label if t!=-100] for label in labels]
    
    str_preds = [[label_names[p] for p,t in zip(pred,targ) if t != -100] for pred,targ in zip(preds,labels)]
    
    the_metrics = metric.compute(predictions=str_preds, references=str_labels)
    return {
        'precision': the_metrics['overall_precision'],
        'recall': the_metrics['overall_recall'],
        'f1': the_metrics['overall_f1'],
        'accuracy': the_metrics['overall_accuracy'],
    }
#%%
id2label = {k:v for k,v in enumerate(label_names)}
label2id = {v:k for k,v in id2label.items()}

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
        )

from transformers import Trainer, TrainingArguments
training_arguments = TrainingArguments(
    "distilbert-finetuned-ner",
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3
)
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

                                                   
# %%
trainer.train()
# %%
trainer.save_model("my_saved_model")
from transformers import pipeline
ner = pipeline("token-classification",
         model="my_saved_model",
         aggregation_strategy='simple',
         device=0,
        )
#%%

s = "Buzz Light-year is the best character in Toy Story"
ner(s)
