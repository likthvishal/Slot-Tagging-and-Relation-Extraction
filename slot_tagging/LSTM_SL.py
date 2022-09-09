import pandas as pd
import numpy as np
from collections import Counter
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,make_scorer
seed=1
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
train_df=pd.read_csv('./train_data.csv')
train_df.head()

test_df=pd.read_csv('./test_data.csv')
test_df.head()

x_test=test_df['utterances'].values

x_train, x_val, y_train, y_val = train_test_split(train_df['utterances'].values, train_df['IOB Slot tags'].values, 
                                test_size=0.2, random_state=1, shuffle=True)

all_sents=x_train.tolist()+x_val.tolist()+x_test.tolist()
vocab=[]
for i in all_sents:
    vocab+=i.split()
vocab=set(vocab)
print ('Vocab length:',len(vocab))

vocab=['<PAD>']+list(vocab)
word2idx={vocab[i]:i for i in range(len(vocab))}
idx2word={i:vocab[i] for i in range(len(vocab))}
assert len(word2idx)==len(idx2word)

all_tags=y_train.tolist()+y_val.tolist()
tags=[]
for i in all_tags:
    tags+=i.split()
tags=set(tags)
print ('tags length:',len(tags))

tags=['<PAD>']+list(tags)
tag2idx={tags[i]:i for i in range(len(tags))}
idx2tag={i:tags[i] for i in range(len(tags))}
assert len(tag2idx)==len(idx2tag)

def vectorize_vocab(word2idx,*argv):
    groups=[]
    for sents in argv[0]:
        groups.append([[word2idx[j] for j in i.split()] for i in sents])
    return groups

x_train,x_val,x_test=vectorize_vocab(word2idx,[x_train,x_val,x_test])
max_len=max([len(i) for i in x_train])

y_train,y_val=vectorize_vocab(tag2idx,[y_train,y_val])
assert max_len==max([len(i) for i in y_train])

from torch.utils.data import Dataset, DataLoader

class MovieData(Dataset):
    def __init__(self, X, y):
        try:
            self.X = torch.tensor(X)
        except ValueError as e:
            self.X = [torch.tensor(i) for i in X]
        self.y = [torch.tensor(i) for i in y]

    def __len__(self):
        return len(self.y)

    def __getitem__(self,index):
        return self.X[index], self.y[index]


def sort_batch(x):
    longest = max([len(i[0]) for i in x])
    s=torch.stack([torch.concat([i[0],torch.zeros(longest-len(i[0])).long()]) for i in x])
    l=torch.stack([torch.concat([i[1],torch.zeros(longest-len(i[1])).long()]) for i in x])
    return s,l

train_dataset = MovieData(X=x_train,y=y_train)
val_dataset = MovieData(X=x_val,y=y_val)

bs=32

train_dataloader = DataLoader(dataset=train_dataset,collate_fn=sort_batch,batch_size=bs,shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset,collate_fn=sort_batch,batch_size=bs,shuffle=True)

# class LSTM_(nn.Module):
#     def __init__(self,vocab_size,embed_dim,num_class,hidden_size,padding_index=0):
#         super(LSTM_,self).__init__()
#         self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_index)
#         self.rnn=nn.LSTM(input_size=embed_dim,hidden_size=hidden_size,num_layers=1,batch_first=True)
#         self.fc1= nn.Linear(hidden_size,num_class)
        
#     def forward(self,x):
#         x=self.emb(x)
#         out,h_n=self.rnn(x)
#         print (out.shape)
#         h1=self.fc1(out)
#         print (h1.shape)
#         # h2=F.log_softmax(h1,dim=-1)
#         h2=h1.view(-1, h1.shape[-1])
#         return h2

class LSTM_1(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_class,hidden_size,padding_index=0):
        super(LSTM_1,self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_index)
        self.dropout = nn.Dropout(0.2)
        self.rnn=nn.LSTM(input_size=embed_dim,hidden_size=hidden_size,num_layers=2,batch_first=True,bidirectional=True)
        
        self.fc1= nn.Linear(hidden_size*2,num_class)
        
    def forward(self,x):
        x=self.dropout(self.emb(x))
        out,h_n=self.rnn(x)
        # print (out.shape)
        h1=self.fc1(out)
        # print (h1.shape)
        # h2=F.log_softmax(h1,dim=-1)
        h2=h1.view(-1, h1.shape[-1])
        return h2

embed_size=300
hidden_size=64
# clf = LSTM_1(vocab_size=len(vocab),embed_dim=embed_size,num_class=len(tags),hidden_size=hidden_size)
# loss_func = nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()


clf = LSTM_1(vocab_size=len(vocab),embed_dim=embed_size,num_class=len(tags),hidden_size=hidden_size)
learning_rate=0.001
num_epochs=15
optimizer = optim.Adam(clf.parameters(), lr=learning_rate)
losses = []
train_losses,train_acc = [],[]
val_losses, val_acc=[],[]
min_loss=1000
total_step=len(train_dataloader)

for epoch in range(num_epochs):
    train_loss,correct=0,0
    total_tags=0
    total_val_tags=0
    for i,(X,y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        # print (X.shape,y.shape)
        y_pred = clf(X)
        # print (y_pred.shape,y.shape)
        # break
    # break
#         break
        y=y.view(-1)
        loss = loss_func(y_pred, y)
        train_loss+=loss.item()

        y_pred=torch.argmax(y_pred,dim=1)
        y_pred,y=y_pred.flatten(),y.flatten()
        correct+= (y_pred==y).sum().item()
        total_tags+=y.shape[0]
        
        loss.backward()
        optimizer.step()
        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        # break
    train_loss=round(train_loss/len(val_dataloader),3)
    acc=round(100*correct/total_tags,3)
    train_losses.append(train_loss)
    train_acc.append(acc)
    print (f"Train loss: {train_loss} Train accuracy: {acc}")  

    val_loss,correct=0,0
    for i,(X,y) in enumerate(val_dataloader):
        y_pred = clf(X)
        y=y.view(-1)
        loss = loss_func(y_pred, y)
        val_loss+= round(loss.item(),3)

        y_pred=torch.argmax(y_pred,dim=1)
        # print (y_pred.shape)
        y_pred,y=y_pred.flatten(),y.flatten()
        # print (y[0:5],y_pred[0:5])
        # print ((y_pred==y).sum().item(),len(y),total_val_tags)
        correct+= (y_pred==y).sum().item()
        total_val_tags+=len(y)
        
    val_loss=round(val_loss/len(val_dataloader),3)

    if(val_loss<min_loss):
        print ('\nSaving best model...')
        min_loss=val_loss
        torch.save(clf.state_dict(), './best_model.pt')

    val_losses.append(val_loss)
    acc=round(100*correct/total_val_tags,3)
    val_acc.append(acc)
    print (f'Valid loss: {val_loss} Valid acc: {acc}')
    print ('\n'+'='*20)

preds=[]
preds_sent=[]
for i,X in enumerate(x_test):
    # print (torch.tensor(X).unsqueeze(0))
    y_pred = clf(torch.tensor(X).unsqueeze(0))
    y_pred=torch.argmax(y_pred,dim=1)
    # print (y_pred.detach)
    preds_sent.append(y_pred.detach().numpy().tolist())
    preds+=y_pred.detach().numpy().tolist()

preds=[idx2tag[i] for i in preds]
preds_sent=[(' ').join([idx2tag[j] for j in i]) for i in preds_sent]

test_sents=test_df.values.tolist()

preds_ = pd.DataFrame(zip(range(len(preds)),preds),columns=['Id','Predicted'])
preds_.head()

preds_.to_csv('./preds.csv',index=None)
