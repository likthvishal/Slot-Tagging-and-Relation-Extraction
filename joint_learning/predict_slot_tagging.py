import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_metric
from typing import Optional
import re
from dataclasses import dataclass
from datasets import Dataset
from collections import Counter
import random
import os
random.seed(1234)
import warnings
warnings.filterwarnings("ignore")


os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'
os.environ['TRANSFORMERS_CACHE'] = '/data/users/kartik/hfcache/'
os.environ['HF_HOME'] = '/data/users/kartik/hfcache/'
os.environ["WANDB_DISABLED"] = "true"

model_checkpoint = ["distilbert-base-uncased",'bert-base-uncased'][1]
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

df = pd.read_csv('./hw1_train.csv')
df.head(3)

train_df=df.sample(frac=0.8) #random state is a seed value
val_df=df.drop(train_df.index)

full_dataset = False
if full_dataset:
    train_df = df

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print ('Train dataset length: ',len(train_df))
print ('Valid dataset length: ',len(val_df))

train_df['utterances'] = train_df['utterances'].apply(lambda x:x.split())
# train_df['utterances'] = train_df['utterances'].apply(lambda x:['BOS']+x.split()+['EOS'])
train_df['IOB Slot tags'] = train_df['IOB Slot tags'].apply(lambda x:x.split())
# train_df['IOB Slot tags'] = train_df['IOB Slot tags'].apply(lambda x:['BOS']+x.split()+['EOS'])

val_df['utterances'] = val_df['utterances'].apply(lambda x:x.split())
# val_df['utterances'] = val_df['utterances'].apply(lambda x:['BOS']+x.split()+['EOS'])
val_df['IOB Slot tags'] = val_df['IOB Slot tags'].apply(lambda x:x.split())
# val_df['IOB Slot tags'] = val_df['IOB Slot tags'].apply(lambda x:['BOS']+x.split()+['EOS'])

print (train_df.head(3))

train_df = train_df.rename({'utterances': 'tokens'},axis=1)
val_df = val_df.rename({'utterances': 'tokens'},axis=1)

train_dataset = Dataset.from_pandas(train_df[['tokens','IOB Slot tags']])
val_dataset = Dataset.from_pandas(val_df[['tokens','IOB Slot tags']])
print (f'Train dataset: {train_dataset}, Val dataset: {val_dataset}')

labels_freq = Counter([j for i in df['IOB Slot tags'].apply(lambda x:x.split()).tolist() for j in i])
labels_list = list(labels_freq.keys())
# labels_list = list(labels_freq.keys()) + ['BOS','EOS']
labels_to_integer = {labels_list[i]:i for i in range(len(labels_list))}


batch_size = 16

class SeqData(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index]

def collate_labels(batch):
    label_all_tokens = True
    tokenized_inputs = tokenizer(batch['tokens'], truncation=True, padding=True, 
                                 is_split_into_words=True, return_offsets_mapping=True)

    labels = []
    for i, label in enumerate(batch['IOB Slot tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels_to_integer[label[word_idx]]) 
            else:
                label_ids.append(labels_to_integer[label[word_idx]] if label_all_tokens else -100) # if a word is tokenized into multiple subwords, assign the main_word tag
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


train_tokenized_datasets = train_dataset.map(collate_labels, batched=True, batch_size=batch_size)
val_tokenized_datasets = val_dataset.map(collate_labels, batched=True, batch_size=batch_size)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(labels_list))

args = TrainingArguments(
    f"{model_checkpoint}-ner",
    evaluation_strategy = "epoch",
    logging_steps=100,
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    num_train_epochs=4,
    weight_decay=1e-2,
    save_strategy='epoch',
)
print ('args',args)
data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[labels_list[p] for (p, l) in zip(prediction, label) if l!=-100] for prediction, label in zip(predictions, labels)]
    true_labels = [[labels_list[l] for (p, l) in zip(prediction, label) if l!=-100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels, suffix=False)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
    

model_checkpoint_load = f'checkpoints/{model_checkpoint}-ner/checkpoint-152_best'
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint_load, num_labels=len(labels_list))
trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=val_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

test_df = pd.read_csv('./hw1_test.csv')
# test_df = pd.read_csv('./hw1_val.csv')
test_df['utterances'] = test_df['utterances'].apply(lambda x:x.split())
# test_df['utterances'] = test_df['utterances'].apply(lambda x:['BOS']+x.split()+['EOS'])

print ('Test dataset length: ',len(test_df))
test_df = test_df.rename({'utterances': 'tokens'},axis=1)
test_dataset = Dataset.from_pandas(test_df)
print (test_dataset[0])

def collate_test_labels(batch):
    return tokenizer(batch['tokens'], truncation=True, padding=True, 
                                 is_split_into_words=True,return_offsets_mapping=True)
test_tokenized_datasets = test_dataset.map(collate_test_labels)
test_tokenized_datasets[0]

predictions = trainer.predict(test_tokenized_datasets).predictions
preds = np.argmax(predictions,axis=2)
test_preds = []
preds = [[labels_list[i] for i in j] for j in preds]
for i in range(len(test_tokenized_datasets)):
    off_maps = np.array(test_tokenized_datasets[i]['offset_mapping'])
    cur_pred = np.array(preds[i])[:len(off_maps)]
    off_preds = cur_pred[(off_maps[:,0] == 0) & (off_maps[:,1] != 0)]
    test_preds.append(off_preds)

# out_preds = [j for i in test_preds for j in i]
out_preds = [j for i in test_preds for j in i[1:-1]]
print ('Len output',len(out_preds))
print ('Saving...')
pd.DataFrame(zip(range(len(out_preds)), out_preds),columns=['Id','Predicted']).to_csv('preds/bert_st.csv',index=None)
