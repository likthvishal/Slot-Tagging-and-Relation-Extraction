import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
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
import numpy as np


from sklearn.metrics import accuracy_score, classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'
os.environ['TRANSFORMERS_CACHE'] = '/data/users/kartik/hfcache/'
os.environ['HF_HOME'] = '/data/users/kartik/hfcache/'
os.environ["WANDB_DISABLED"] = "true"

df = pd.read_csv('./hw1_train.csv')
df.head(3)

df['utterances'] = df['utterances'].apply(lambda x:x.split())
df['IOB Slot tags'] = df['IOB Slot tags'].apply(lambda x:x.split())

df['utterances'] = df['utterances'].apply(lambda x:['BOS']+x+['EOS'])
df['iob_tags'] = df['IOB Slot tags'].apply(lambda x:['BOS']+x+['EOS'])
df['core_relations'] = df['Core Relations'].astype('str').apply(lambda x:('#').join(x.split()))
df = df.drop(['IOB Slot tags','Core Relations'],axis=1)

train_df=df.sample(frac=0.8) #random state is a seed value
val_df=df.drop(train_df.index)

print(train_df.head(3))

slot_vocab = list(set([j for i in df['iob_tags'].tolist() for j in i]))
print ('Slot vocab',slot_vocab)

intent_vocab = set(df['core_relations'].tolist())
print ('Intent vocab',len(intent_vocab),intent_vocab)

train_sentences = train_df.utterances.tolist()
train_slots = train_df.iob_tags.tolist()
train_intents = train_df.core_relations.tolist()

val_sentences = val_df.utterances.tolist()
val_slots = val_df.iob_tags.tolist()
val_intents = val_df.core_relations.tolist()

print (len(train_sentences),len(train_slots),len(train_intents))
print (len(val_sentences),len(val_slots),len(val_intents))
print (train_sentences[0],train_slots[0],train_intents[0])

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import Dropout
from transformers import BertModel, BertTokenizer, BertTokenizerFast

class ParserModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 num_slot_labels: int):
        super(ParserModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)
        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor = None,
                slot_labels: torch.tensor = None
                ):
      
        last_hidden_states, pooler_output = self.bert_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                   return_dict=False)
        
        #print(self.bert_model.config)

        # intent_logits = self.intent_classifier(pooler_output)
        # slot_logits = self.slot_classifier(last_hidden_states)
        intent_logits = self.intent_classifier(self.dropout(pooler_output))
        slot_logits = self.slot_classifier(self.dropout(last_hidden_states))

        loss_fct = CrossEntropyLoss()
        # Compute losses if labels provided
        if intent_label is not None:
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        if slot_labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        return intent_logits, slot_logits, slot_loss, intent_loss
    

# 1. For mapping slots and intents between ints and string
slot2id = {slot:id for id,slot in enumerate(slot_vocab)}
id2slot = {id:slot for slot,id in slot2id.items()}

intent2id = {intent:id for id,intent in enumerate(intent_vocab)}
id2intent = {id:intent for intent,id in intent2id.items()}


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)


# 4. pads to longest sequence, truncates to maximum allowed length by model, 
# Gives encodings, token type_ids -> https://huggingface.co/transformers/glossary.html#token-type-ids and attention_masks
# and offset_mappings
train_encodings = tokenizer(train_sentences, return_offsets_mapping=True,is_split_into_words=True, padding=True, truncation=True)
val_encodings = tokenizer(val_sentences, return_offsets_mapping=True,is_split_into_words=True, padding=True, truncation=True)

# 5. Encoding labels

def encode_labels(tags, encodings, mapping):
    labels = [[mapping[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

def encode_intents(intents, mapping):
    labels = [mapping[intent] for intent in intents]
    return labels

    
train_slot_labels = encode_labels(train_slots, train_encodings, slot2id)
train_intent_labels =  encode_intents(train_intents, intent2id)

val_slot_labels = encode_labels(val_slots, val_encodings, slot2id)
val_intent_labels =  encode_intents(val_intents, intent2id)

# 6. Make your dataset
import torch

class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, slot_labels, intent_labels):
        self.encodings = encodings
        self.slot_labels = slot_labels
        self.intent_labels = intent_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['slot_labels'] = torch.tensor(self.slot_labels[idx])
        item['intent_labels'] = torch.tensor(self.intent_labels[idx])
        return item

    def __len__(self):
        return len(self.slot_labels)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping") # we don't want to pass this to the model
train_dataset = MovieDataset(train_encodings, train_slot_labels, train_intent_labels)
val_dataset = MovieDataset(val_encodings, val_slot_labels, val_intent_labels)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# # 109,603,742, 109M trainable params compared to 65M with DistilBERT in previous section example.

# 8. Training
from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

best_epoch = 0
def train_step(model, optim, train_loader, val_loader, num_epochs=5):
    global best_epoch
    
    model.train()

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print (f'------- Epoch:{epoch} -------')
        total_slot_loss, total_intent_loss = 0,0


        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            intent_labels = batch['intent_labels'].to(device)
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids, 
                            slot_labels=slot_labels,
                            intent_label=intent_labels)
            slot_loss, intent_loss = outputs[2],outputs[3]


            total_slot_loss+=slot_loss.item()
            total_intent_loss+=intent_loss.item()

            slot_loss.backward(retain_graph=True) #need to retain_graph  when working with multiple losses
            intent_loss.backward()
            optim.step()

    
        print (f'Train slot loss {total_slot_loss/len(train_loader)} train intent loss: {total_intent_loss/len(train_loader)}')
        
        train_slot_acc, train_intent_acc = evaluate_validation(model,train_loader)
        print (f'Train slot acc {train_slot_acc} train intent acc: {train_intent_acc}')
        torch.save(model.state_dict(), f'checkpoints/multitask/bert/epoch_{epoch}.model')
        
        val_slot_acc, val_intent_acc = evaluate_validation(model,val_loader)
        val_slot_loss, val_intent_loss = val_step(model,val_loader=val_loader, return_outputs=False)
        print (f'Val slot loss {val_slot_loss}, Val intent loss {val_intent_loss} best epoch {best_epoch}')
        print (f'Train slot acc {val_slot_acc} train intent acc: {val_intent_acc}')

        if val_slot_loss < best_val_loss:
            best_val_loss = val_slot_loss
            best_epoch = epoch
        model.train()

    return (total_slot_loss/len(train_loader), total_intent_loss/len(train_loader))

def evaluate_accuracy(intent_logits, slot_logits, intents_true, slots_true):

        # slots
        probability_testue = torch.softmax(slot_logits,dim=2)
        idxs = torch.argmax(probability_testue,dim=2)[0]
        intent_probability_testue = torch.softmax(intent_logits,dim=1)
        intent_idxs = torch.argmax(intent_probability_testue,dim=1)

        slots_pred = [idxs[id].item() for id in range(len(idxs)) if slots_true[0][id]!=-100]
        slots_true = [i.item() for i in slots_true[0] if i!=-100]
        # print (intents_true, intent_idxs,intents_true==intent_idxs, slots_true==slots_pred)
        
        return int(intents_true==intent_idxs), int(slots_true==slots_pred)

def evaluate_validation(model,val_loader):
    model.eval()
    val_slot_acc, val_intent_acc = 0,0

    for batch in val_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        slot_labels = batch['slot_labels'].to(device)
        intent_labels = batch['intent_labels'].to(device)
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids, 
                        slot_labels=slot_labels,
                        intent_label=intent_labels)
        slot_loss, intent_loss = outputs[2],outputs[3]

        sa, ia = evaluate_accuracy(outputs[0],outputs[1], intent_labels, slot_labels)
        val_slot_acc+=sa
        val_intent_acc+=ia
    return val_slot_acc/len(val_loader), val_intent_acc/len(val_loader)



def val_step(model, val_loader, return_outputs=True):
    model.eval()

    val_slot_loss, val_intent_loss = 0,0
    val_slot_preds, val_intent_preds = [],[]

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        slot_labels = batch['slot_labels'].to(device)
        intent_labels = batch['intent_labels'].to(device)
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids, 
                        slot_labels=slot_labels,
                        intent_label=intent_labels)
        intent_logits, slot_logits =  outputs[0], outputs[1]
        slot_loss, intent_loss = outputs[2], outputs[3]
        if return_outputs:
        # slots
            probability_value = torch.softmax(slot_logits,dim=2)
            idxs = torch.argmax(probability_value,dim=2)[0]
            # print (slot_labels,idxs)
            #intent
            intent_probability_value = torch.softmax(intent_logits,dim=1)
            intent_idxs = torch.argmax(intent_probability_value,dim=1)

            slot_prediction = [id2slot[idxs[id].item()] for id in range(len(idxs)) if slot_labels[0][id]!=-100]
            intent_prediction = [id2intent[id.item()] for id in intent_idxs]
            val_slot_preds.append(slot_prediction)
            val_intent_preds.append(intent_prediction)

        val_slot_loss+= slot_loss.item()
        val_intent_loss+=intent_loss.item()
    
    val_slot_loss /= len(val_loader)
    val_intent_loss /=len(val_loader)

    return (val_slot_loss,val_intent_loss, val_slot_preds, val_intent_preds) if return_outputs else (val_slot_loss,val_intent_loss)



dropout = 0.2
num_intent_labels = len(intent_vocab)
num_slot_labels = len(slot_vocab)

model = ParserModel(model_name_or_path='bert-base-uncased',
                    dropout=dropout, 
                    num_intent_labels=num_intent_labels, 
                    num_slot_labels=num_slot_labels,
                   )
print (count_parameters(model))
model.load_state_dict(torch.load(f'checkpoints/multitask/bert/epoch_4.model'))


model.to(device)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 'W' stands for 'Weight Decay fix"
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optim = AdamW(model.parameters(), lr=5e-5)
train_step(model=model, optim=optim, train_loader=train_loader, val_loader=val_loader, num_epochs=5)

print (f'Loading checkpoints/multitask/bert/epoch_{best_epoch}.model')
model.load_state_dict(torch.load(f'checkpoints/multitask/bert/epoch_{best_epoch}.model'))
model.eval()
model.to(device)

test_df = pd.read_csv('./hw1_test.csv')#[:3]
test_df['utterances'] = test_df['utterances'].apply(lambda x:x.split())
test_df['utterances'] = test_df['utterances'].apply(lambda x:['BOS']+x+['EOS'])

test_sentences = test_df.utterances.tolist()
test_slots = [['BOS']+['O']*(len(i)-2)+['EOS'] for i in test_sentences]
test_encodings = tokenizer(test_sentences, return_offsets_mapping=True,is_split_into_words=True, padding=True, truncation=True)
test_slot_labels = encode_labels(test_slots, test_encodings, slot2id)

class MovieTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, dummy_slot_labels):
        self.encodings = encodings
        self.slot_labels = dummy_slot_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['slot_labels'] = torch.tensor(self.slot_labels[idx])
        # item['intent_labels'] = torch.tensor(self.intent_labels[idx])
        return item

    def __len__(self):
        return len(self.slot_labels)

# test_encodings.pop("offset_mapping") # we don't want to pass this to the model
test_dataset = MovieTestDataset(test_encodings, test_slot_labels)

# # 10. Inference, you have to do inference to generate your prediction.txt for hw1
model.eval()
model.to(device)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)# reusing training set lol
losses = []

test_slot_preds, test_intent_preds = [],[]

for batch in test_loader:
  input_ids = batch['input_ids'].to(device)
  attention_mask = batch['attention_mask'].to(device)
  token_type_ids = batch['token_type_ids'].to(device)
  slot_labels = batch['slot_labels'].to(device)
  # intent_labels = batch['intent_labels'].to(device)
  outputs = model(input_ids=input_ids, 
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids)
  intent_logits, slot_logits =  outputs[0], outputs[1]

  # slots
  probability_testue = torch.softmax(slot_logits,dim=2)
  idxs = torch.argmax(probability_testue,dim=2)[0]
  # print (idxs,slot_labels)
  #intent
  intent_probability_testue = torch.softmax(intent_logits,dim=1)
  intent_idxs = torch.argmax(intent_probability_testue,dim=1)
  #true = [id2tag[id.item()] for id in labels[0]]
  # for id in range(len(idxs)):
    # print (slot_labels[id],idxs[id])
  slot_prediction = [id2slot[idxs[id].item()] for id in range(len(idxs)) if slot_labels[0][id]!=-100]
  intent_prediction = [id2intent[id.item()] for id in intent_idxs]
  test_slot_preds.append(slot_prediction)
  test_intent_preds.append(intent_prediction)


test_sp, test_ip = [],[]
for _slot_preds, _int_preds in zip(test_slot_preds,test_intent_preds):
    _slot_preds = (' ').join(_slot_preds[1:-1])
    _int_preds = _int_preds[0]
    if _int_preds=='nan': _int_preds=''
    elif '#' in _int_preds: _int_preds = _int_preds.replace('#',' ')
    test_sp.append(_slot_preds)
    test_ip.append(_int_preds)

out_preds = [j for i in test_sp for j in i.split()]
print ('Len output',len(out_preds))

print ('Saving predictions...')
pd.DataFrame(zip(range(len(out_preds)), out_preds),columns=['Id','Predicted']).to_csv('preds/bert_st.csv',index=None)

pd.DataFrame(zip(range(len(test_ip)), test_ip),columns=['Id','Predicted']).to_csv('preds/bert_cr.csv',index=None)

with open('preds/jointbert_preds.txt','w') as f:
    for _slot_preds,_int_preds in zip(test_sp,test_ip):
        f.write(_slot_preds+'\t'+_int_preds+'\n')
