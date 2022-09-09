import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_metric
from typing import Optional
import re
from dataclasses import dataclass
from collections import Counter
import os
import warnings
warnings.filterwarnings("ignore")
# from tqdm import tqdm
from tqdm.notebook import tqdm
import random

seed_val = 1234
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

os.environ['CUDA_VISIBLE_DEVICES'] = '3,5'
os.environ['TRANSFORMERS_CACHE'] = '/data/users/kartik/hfcache/'
os.environ['HF_HOME'] = '/data/users/kartik/hfcache/'
os.environ["WANDB_DISABLED"] = "true"


df = pd.read_csv('./hw1_train.csv')
df.head(3)

idx_to_cr = list(set(df['Core Relations'].astype('str').tolist()))
label_dict = {val:idx for idx,val in enumerate(idx_to_cr)}
label_dict_inverse = {idx:val for idx,val in enumerate(idx_to_cr)}

df['core_relations'] = df['Core Relations'].astype('str').apply(lambda x:label_dict[x])

df = df.drop(columns = ['IOB Slot tags','Core Relations'])

train_df=df.sample(frac=0.9) #random state is a seed value
val_df=df.drop(train_df.index)
print ('Train dataset length: ',len(train_df))
print ('Valid dataset length: ',len(val_df))

from torch.utils.data import TensorDataset

model_checkpoint = ['xlm-roberta-base', 'distilbert-base-uncased','bert-base-uncased','albert-base-v2'][2]


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

encoded_data_train = tokenizer.batch_encode_plus(
    train_df.utterances.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    val_df.utterances.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(train_df.core_relations.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(val_df.core_relations.values)   

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

print ('len(dataset_train), len(dataset_val)',len(dataset_train), len(dataset_val))

num_class = len(idx_to_cr)
print (f'Num of classes: {num_class}')

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                      num_labels= num_class,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 4

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)


from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)                                   
epochs = 1

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)                  
from sklearn.metrics import f1_score, accuracy_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat),f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')      

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)                                            
def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

best_val_loss,best_epoch = float('inf'),1
patience, max_patience = 0,3
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    print (f'Saving checkpoints/singlelabel/{model_checkpoint}/epoch_{epoch}.model')
    torch.save(model.state_dict(), f'checkpoints/singlelabel/{model_checkpoint}/epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    train_loss, predictions, true_vals = evaluate(dataloader_train)
    train_acc, train_f1 = f1_score_func(predictions, true_vals)

    tqdm.write(f'Training loss: {loss_train_avg}')
    tqdm.write(f'Training Accuracy: {train_acc}')
    tqdm.write(f'Training F1: {train_f1}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_acc, val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_acc}')
    tqdm.write(f'Val F1: {val_f1}')

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    if val_loss < best_val_loss:
        patience = 0
        best_epoch = epoch
        best_val_loss = val_loss
    else:
        patience +=1

    print ('Patience',patience,'best_loss',best_val_loss,'val loss',val_loss)

    if patience == max_patience:
        break



model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                      num_labels=num_class,
                                                      output_attentions=False,
                                                      output_hidden_states=False)


print (f'Loading checkpoints/singlelabel/{model_checkpoint}/epoch_{best_epoch}.model')
model.load_state_dict(torch.load(f'checkpoints/singlelabel/{model_checkpoint}/epoch_{best_epoch}.model'))
model.eval()
model.to(device)

test_df = pd.read_csv('./hw1_test.csv')#[:3]
# test_df = pd.read_csv('./hw1_val.csv')#[:3]
test_df['core_relations']=[0]*len(test_df)
test_df        


encoded_data_test = tokenizer.batch_encode_plus(
    test_df.utterances.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)


input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(test_df.core_relations.values)

dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)


dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=batch_size)

_, predictions, true_vals = evaluate(dataloader_test)

preds = [label_dict_inverse[i] for i in np.argmax(predictions, axis=1).flatten()]
preds = [i if i!='nan' else 'none' for i in preds ]

pd.DataFrame(zip(range(len(test_df)), preds),columns=['Id','Predicted']).to_csv('preds/bert_cr.csv',index=None)
