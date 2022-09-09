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

vocab=['<START>','<STOP>','<PAD>']+list(vocab)
word2idx={vocab[i]:i for i in range(len(vocab))}
idx2word={i:vocab[i] for i in range(len(vocab))}
assert len(word2idx)==len(idx2word)

all_tags=y_train.tolist()+y_val.tolist()
tags=[]
for i in all_tags:
    tags+=i.split()
tags=set(tags)
print ('tags length:',len(tags))

tags=['<START>','<STOP>','<PAD>']+list(tags)
tag2idx={tags[i]:i for i in range(len(tags))}
idx2tag={i:tags[i] for i in range(len(tags))}
assert len(tag2idx)==len(idx2tag)

def vectorize_vocab(word2idx,*argv):
    groups=[]
    for sents in argv[0]:
        groups.append([[word2idx['<START>']]+[word2idx[j] for j in i.split()]+[word2idx['<STOP>']] for i in sents])
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
    # l=torch.stack([i[1] for i in x])
    # print (l)
    return s,l


train_dataset = MovieData(X=x_train,y=y_train)
val_dataset = MovieData(X=x_val,y=y_val)

bs=32

train_dataloader = DataLoader(dataset=train_dataset,collate_fn=sort_batch,batch_size=bs,shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset,collate_fn=sort_batch,batch_size=bs,shuffle=True)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell
        
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs


INPUT_DIM = len(vocab)
OUTPUT_DIM = len(tags)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 64
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

device='cpu'
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

loss_func = nn.CrossEntropyLoss()
learning_rate=0.01
num_epochs=32
losses = []
train_losses,train_acc = [],[]
val_losses, val_acc=[],[]
min_loss=1000
total_step=len(train_dataloader)
optimizer = optim.Adam(model.parameters())

for epoch in range(num_epochs):
    train_loss,correct=0,0
    total_tags=0
    total_val_tags=0
    for i,(X,y) in enumerate(train_dataloader):
        
        optimizer.zero_grad()
        X,y=X.transpose(0,1), y.transpose(0,1)
        # print (X.shape,y.shape)
        
        y_pred = model(X, y)
        # print (y_pred.shape)
        assert y_pred.shape[0]==y.shape[0]
        
        y_pred_dim = y_pred.shape[-1]
        y_pred = y_pred[1:].view(-1, y_pred_dim)
        y = y[1:].reshape(-1)
        # print (output.shape,y.shape)
        loss = loss_func(y_pred, y)
        
        y_pred=torch.argmax(y_pred,dim=1)
        y_pred,y=y_pred.flatten(),y.flatten()
        correct+= (y_pred==y).sum().item()
        total_tags+=y.shape[0]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        train_loss+=loss.item()
        
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
        X,y=X.transpose(0,1), y.transpose(0,1)
        y_pred = model(X, y)

        
        assert y_pred.shape[0]==y.shape[0]
        
        y_pred_dim = y_pred.shape[-1]
        y_pred = y_pred[1:].view(-1, y_pred_dim)
        y = y[1:].reshape(-1)
        
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
        torch.save(model.state_dict(), './best_model.pt')

    val_losses.append(val_loss)
    acc=round(100*correct/total_val_tags,3)
    val_acc.append(acc)
    print (f'Valid loss: {val_loss} Valid acc: {acc}')
    print ('\n'+'='*20)

model.eval()
preds=[]
preds_sent=[]
for i,X in enumerate(x_test):
    # print (X)
    inp=torch.tensor(X).unsqueeze(0).transpose(0,1)
    y_pred = model(inp,inp,0)
    y_pred_dim = y_pred.shape[-1]
    y_pred = y_pred[1:].view(-1, y_pred_dim)
    y_pred=torch.argmax(y_pred,dim=1).flatten()
    # print (y_pred)
    preds_sent.append(y_pred.detach().numpy().tolist())
    preds_=y_pred.detach().numpy().tolist()
    # print (len(preds_),len(X)-1)
    preds+=preds_
    # break
print (preds)
print (len(preds))

#  if idx2tag[i]!='<STOP>'
preds=[idx2tag[i] for i in preds]
preds_sent=[(' ').join([idx2tag[j] for j in i]) for i in preds_sent]

test_sents=test_df.values.tolist()

preds_ = pd.DataFrame(zip(range(len(preds)),preds),columns=['Id','Predicted'])
preds_.head()

preds_.to_csv('./preds.csv',index=None)
