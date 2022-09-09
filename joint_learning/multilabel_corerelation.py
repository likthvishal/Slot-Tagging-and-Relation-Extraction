import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_metric
from typing import Optional
import re
from dataclasses import dataclass
from collections import Counter
import random
import os
random.seed(1234)
import warnings
warnings.filterwarnings("ignore")
# from tqdm import tqdm
from tqdm.notebook import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TRANSFORMERS_CACHE'] = '/data/users/kartik/hfcache/'
os.environ['HF_HOME'] = '/data/users/kartik/hfcache/'
os.environ["WANDB_DISABLED"] = "true"


df = pd.read_csv('./hw1_train.csv')
df.head(3)

df['Core Relations'] = df['Core Relations'].astype('str').apply(lambda x:x.split())

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
onehot_df = pd.DataFrame(mlb.fit_transform(df['Core Relations'].tolist()),columns=mlb.classes_)
df=pd.concat([df,onehot_df],axis=1)
df = df.drop(columns = ['IOB Slot tags','Core Relations'])
df['core_relations'] = df.drop('utterances',axis=1).values.tolist()
df = df[['utterances','core_relations']]


train_df=df.sample(frac=0.8) #random state is a seed value
val_df=df.drop(train_df.index)
print ('Train dataset length: ',len(train_df))
print ('Valid dataset length: ',len(val_df))

from torch.utils.data import TensorDataset

# from transformers import DistilBertTokenizer,DistilBertForSequenceClassification, XLMForSequenceClassification

model_checkpoint = ['xlm-roberta-base', 'distilbert-base-uncased','bert-base-uncased','albert-base-v2'][3]


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

encoded_data_train = tokenizer(train_df.utterances.tolist(), truncation=True, padding=True, 
                                 is_split_into_words=False, return_tensors='pt')

encoded_data_val = tokenizer(val_df.utterances.tolist(), truncation=True, padding=True, 
                                 is_split_into_words=False, return_tensors='pt')

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(train_df.core_relations.tolist())

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(val_df.core_relations.tolist())

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
print ('len(dataset_train), len(dataset_val)', len(dataset_train), len(dataset_val))

num_class = len(labels_train[0])
print (f'Num of classes: {num_class}')
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
#                                                       num_labels= num_class)

class CustomBERTClass(torch.nn.Module):
    def __init__(self):
        super(CustomBERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_checkpoint)
        # self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_class)

    def forward(self, input_ids, attention_mask, labels):
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        # print (output.last_hidden_state.shape)
        if model_checkpoint!='bert-base-uncased':
            pooler = output.last_hidden_state[:,-1,:]
        else:
            pooler = output.pooler_output

        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = CustomBERTClass()

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 16

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)


from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-4, 
                  eps=1e-8)

epochs = 20

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


seed_val = 1234
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


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
            
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(outputs, batch[2].float())
        loss_val_total += loss.item()

        logits = outputs.detach().cpu().numpy()
        logits = (logits>0).astype('int')
        
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

best_val_loss = float('inf')
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
        
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(outputs, batch[2].float())
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
    print (f'Saving checkpoints/multilabel/{model_checkpoint}/epoch_{epoch}.model')
    torch.save(model.state_dict(), f'checkpoints/multilabel/{model_checkpoint}/epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    train_loss, predictions, true_vals = evaluate(dataloader_train)
    train_acc = np.sum((true_vals==predictions).all(1))/len(predictions)

    tqdm.write(f'Training loss: {loss_train_avg}')
    tqdm.write(f'Training Accuracy: {train_acc}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    if val_loss < best_val_loss:
        patience = 0
        best_epoch = epoch
        best_val_loss = val_loss
    else:
        patience +=1

    # inv_label_preds = [np.where(predictions[i,:]==1)[0].tolist() for i in range(len(predictions))]
    # inv_label_true = [np.where(true_vals[i,:]==1)[0].tolist() for i in range(len(predictions))]
    # val_f1 = np.sum((inv_label_true==inv_label_preds).all(1))/len(predictions)

    val_acc = np.sum((true_vals==predictions).all(1))/len(predictions)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'Val Accuracy: {val_acc}')
    print ('Patience',patience,'best_loss',best_val_loss,'val loss',val_loss)

    if patience == max_patience:
        break


print (f'Loading checkpoints/multilabel/{model_checkpoint}/epoch_{best_epoch}.model')
model.load_state_dict(torch.load(f'checkpoints/multilabel/{model_checkpoint}/epoch_{best_epoch}.model'))
model.eval()


_, predictions, true_vals = evaluate(dataloader_validation)


test_df = pd.read_csv('./hw1_test.csv')#[:3]
test_df['core_relations']=[[0]*num_class]*len(test_df)
test_df

encoded_data_test = tokenizer(test_df.utterances.tolist(), truncation=True, padding=True, 
                                 is_split_into_words=False, return_tensors='pt')



input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(test_df.core_relations.tolist())

dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)


dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=batch_size)


def predict_test(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
        

        logits = outputs.detach().cpu().numpy()
        logits = (logits>0).astype('int')
        
        predictions.append(logits)
    
    # loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
            
    return predictions

predictions = predict_test(dataloader_test)

preds = [(' ').join(list(i)) for i in mlb.inverse_transform(predictions)]

preds = [i if i!='nan' else 'none' for i in preds ]
pd.DataFrame(zip(range(len(test_df)), preds),columns=['Id','Predicted']).to_csv('preds/bert_cr.csv',index=None)
