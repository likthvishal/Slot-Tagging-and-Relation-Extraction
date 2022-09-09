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

test_tags=[(' ').join(['O']*len(i.split())) for i in test_df['utterances'].values]
print (len(test_tags))
test_df['IOB Slot tags']=test_tags
test_df.head()

all_df=pd.concat([train_df,test_df],ignore_index=True)
all_df.head()

from collections import defaultdict
wordscount=defaultdict(list)
for i in train_df.values.tolist():
    s,t=i[0].split(),i[1].split()
    for j in range(len(s)):
        wordscount[s[j]].append(t[j])
    
wordscount_={}
for k,v in wordscount.items():
    x=Counter(v).most_common()[0]
    # print (k,x)
    if(len(v)>1):
        wordscount_[k]=x[0]
    else:
        wordscount_[k]='<UNK>'

from tqdm import tqdm
s_id,prevprev_word,prev_word,word,next_word,nextnext_word,prevprev_tag,prev_tag,tag,count=[],[],[],[],[],[],[],[],[],[]
for idx,row in tqdm(all_df.iterrows()):
    u_words,u_tags= row['utterances'].split(),row['IOB Slot tags'].split()
    assert len(u_words)==len(u_tags)
    for i in range(len(u_words)):
        s_id.append(idx)
        if i==0:
            prevprev_word.append('<START2>')
            prev_word.append('<START1>')
            # prevprev_tag.append('<START2>')
            # prev_tag.append('<START1>')
        elif i==1:
            prevprev_word.append('<START1>')
            prev_word.append(u_words[0])
            # prevprev_tag.append('<START1>')
            # prev_tag.append(u_tags[0])
        else:
            prevprev_word.append(u_words[i-2])
            prev_word.append(u_words[i-1])
            # prevprev_tag.append(u_tags[i-2])
            # prev_tag.append(u_tags[i-1])
            
        word.append(u_words[i])
        tag.append(u_tags[i])
        try:count.append(wordscount_[u_words[i]])
        except:count.append('<UNK>')
        try:next_word.append(u_words[i+1])
        except:next_word.append('<STOP>')
        try:nextnext_word.append(u_words[i+2])
        except:nextnext_word.append('<STOP>')
        
            
    
word_df=pd.DataFrame(zip(s_id,prevprev_word,prev_word,word,next_word,nextnext_word,count,tag),columns=['s_id','prevprev_word','prev_word','word','next_word','nextnext_word','count','tag'])
y=word_df['tag'].values
num_train_words=sum(word_df['s_id']<len(train_df))
num_test_words=len(word_df)-num_train_words
word_df=word_df.drop(columns=['s_id','tag','count'])
word_df.head(10)

print ('len(word_df),len(train_df),len(test_df):',len(word_df),len(train_df),len(test_df))

x_test=word_df.values[num_train_words:]
print (x_test.shape)

X,y=word_df.values[:num_train_words],y[:num_train_words]
x_train, x_val, y_train, y_val = train_test_split(X, y, 
                                test_size=0.02, random_state=1, 
                                shuffle=True)
print (x_train.shape,y_train.shape,x_val.shape,y_val.shape)
x_train,x_val,y_train,y_val=x_train.tolist(),x_val.tolist(),y_train.tolist(),y_val.tolist()
x_test=x_test.tolist()

all_sents=word_df.values.flatten().tolist()
vocab=all_sents
vocab=list(set(vocab))
print ('Vocab length:',len(vocab))

# vocab=['<PAD>']+list(vocab)
word2idx={vocab[i]:i for i in range(len(vocab))}
idx2word={i:vocab[i] for i in range(len(vocab))}
assert len(word2idx)==len(idx2word)

tags=y.tolist()
tags=set(tags)
print ('tags length:',len(tags))

tags=list(tags)
tag2idx={tags[i]:i for i in range(len(tags))}
idx2tag={i:tags[i] for i in range(len(tags))}
assert len(tag2idx)==len(idx2tag)

def vectorize_vocab(word2idx,*argv):
    groups=[]
    for sents in argv[0]:
        groups.append([[word2idx[j] for j in i] for i in sents])
    return groups

x_train,x_val,x_test=vectorize_vocab(word2idx,[x_train,x_val,x_test])
max_len=max([len(i) for i in x_train])

def vectorize_tag(tag2idx,*argv):
    groups=[]
    for sents in argv[0]:
        groups.append([tag2idx[j] for j in sents])
    return groups
y_train,y_val=vectorize_tag(tag2idx,[y_train,y_val])
# assert max_len==max([len(i) for i in y_train])

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


train_dataset = MovieData(X=x_train,y=y_train)
val_dataset = MovieData(X=x_val,y=y_val)

bs=32

train_dataloader = DataLoader(dataset=train_dataset,batch_size=bs,shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset,batch_size=bs,shuffle=True)


class LSTM_1(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_class,hidden_size,padding_index=0):
        super(LSTM_1,self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_index)
        self.rnn=nn.LSTM(input_size=embed_dim,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.fc1= nn.Linear(hidden_size,num_class)
        
    def forward(self,x):
        x=self.emb(x)
        out,h_n=self.rnn(x)
        h1=self.fc1(out[:,-1,:])
        return h1

embed_size=200
hidden_size=64
clf = LSTM_1(vocab_size=len(vocab),embed_dim=embed_size,num_class=len(tags),hidden_size=hidden_size)
loss_func = nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()

clf = LSTM_1(vocab_size=len(vocab),embed_dim=embed_size,num_class=len(tags),hidden_size=hidden_size)
learning_rate=0.001
num_epochs=6
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
