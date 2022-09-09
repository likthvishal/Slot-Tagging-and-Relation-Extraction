from load_data import MovieData
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,make_scorer
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import numpy as np
from collections import Counter
import random

from helpers import *
from pipeline import *
from models import *


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

seed=1
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

def process_torch_input(x,model_class):
    if(model_class=='rnn' or model_class=='cnn'):
        return x
    else:
        return x.float()

def vectorize_vocab(word2idx,*argv):
    groups=[]
    for sents in argv[0]:
        groups.append([[word2idx[j] for j in i.split()] for i in sents])
    return groups

def sort_batch(x):
    longest = max([len(i[0]) for i in x])
    s=torch.stack([torch.concat([i[0],torch.zeros(longest-len(i[0])).long()]) for i in x])
    l=torch.stack([i[1] for i in x])
    return s,l

def sort_batch_maxlen(x,maxlen):
    longest = maxlen
    s=torch.stack([torch.concat([i[0],torch.zeros(longest-len(i[0])).long()]) for i in x])
    l=torch.stack([i[1] for i in x])
    return s,l

def classify_fc(args):
    train_df,test_df=read_data(args.train_file,args.test_file)
    model_class=args.model_class
    
    cf=args.classification
    if cf=='mlmc':
        train_df['Core Relations']=train_df['Core Relations'].apply(lambda x: x.split())


    # get_distrib(train_df,test_df) 
    # ---------------------- Preprocessing and Label Encoding -------------------------

    train_df,test_df=clean_data(train_df,test_df) 
    train_df,test_df=process_all_data(args,train_df,test_df)

    print ('Sample Data...')
    print ('-'*20)
    print (train_df.utterances.values[:3])
    print ('-'*20)

    if args.pretrained==True:
        from gensim.models import Word2Vec
        import gensim.downloader
        from gensim.test.utils import common_texts
        print ('Loading word embeddings pretrained....')
        pre_model= gensim.downloader.load('fasttext-wiki-news-subwords-300')

        all_sents=train_df['utterances'].values.tolist()+test_df['utterances'].values.tolist()
        vocab=[]
        for i in all_sents:
            vocab+=i.split()
        vocab=set(vocab)
        print ('vocab length',len(vocab))
        vocab=['<PAD>']+list(vocab)
        word2idx={vocab[i]:i for i in range(len(vocab))}
        idx2word={i:vocab[i] for i in range(len(vocab))}

        embedding_matrix=build_embmat(pre_model,vocab)

    if cf=='mc':
        labs=train_df['Core Relations']
        labels=list(set(labs.values))
        labels_mapping={labels[i]:i for i in range(len(labels))}
        labels_mapping_rev={i:labels[i] for i in range(len(labels))}
        train_df['Core Relations']=train_df['Core Relations'].apply(lambda x:labels_mapping[x])
        labs=train_df['Core Relations']


    else:
        mlb = MultiLabelBinarizer()
        labs = pd.DataFrame(mlb.fit_transform(train_df['Core Relations']),columns=mlb.classes_)
        labels=labs.columns.values
    

    x_train, x_val, y_train, y_val = train_test_split(train_df['utterances'].values, labs.values, 
                                                    test_size=args.test_split, random_state=1, 
                                                    shuffle=True)
                                                    # shuffle=True,stratify=train_df['Core Relations'])
    x_test=test_df.utterances.values
    assert (x_train.shape[0]==y_train.shape[0]); assert(x_val.shape[0]==y_val.shape[0])

    # ---------------------- Vectorize data -------------------------

    if(model_class=='rnn' or model_class=='cnn'):

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
        
        x_train,x_val,x_test=vectorize_vocab(word2idx,[x_train,x_val,x_test])
        max_len=max([len(i) for i in x_train])

    
    else:
        x_train,x_val,x_test=vectorize(args.vectorizer,x_train,[x_train,x_val,x_test]) #convert to linear vectors(tfidf/count)
    
    x_train,y_train=sampling(args,x_train,y_train) #over/undersample distribution
    print ('X Train,Val,Test shape',len(x_train),len(x_val),len(x_test))
    print ('y Train,Val shape',len(y_train),len(y_val))



    # ---------------------- Data Loaders -------------------------

    train_dataset = MovieData(X=x_train,y=y_train)
    val_dataset = MovieData(X=x_val,y=y_val)
    bs=args.bs

    if model_class=='nn':
        train_dataloader = DataLoader(dataset=train_dataset, 
                            batch_size=args.bs,#only 1 for this toy example
                            shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, 
                        batch_size=bs,#only 1 for this toy example
                        shuffle=True)

    if(model_class=='rnn'):
        train_dataloader = DataLoader(dataset=train_dataset,collate_fn=sort_batch,batch_size=args.bs,shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset,collate_fn=sort_batch,batch_size=args.bs,shuffle=True)

    if(model_class=='cnn'):
        train_dataloader = DataLoader(dataset=train_dataset,collate_fn=lambda x: sort_batch_maxlen(x,maxlen=max_len),batch_size=args.bs,shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset,collate_fn=lambda x: sort_batch_maxlen(x,maxlen=max_len),batch_size=args.bs,shuffle=True)

    
    
    # ---------------------- Modelling -------------------------

    learning_rate=args.lr
    num_epochs=args.n_epochs
    
    if model_class=='nn':
        clf = MLP_PretrainedVecs_3(x_train.shape[1],len(labels),512)
    elif model_class=='rnn':
        clf = LSTM_1(vocab_size=len(vocab),embed_dim=args.embed_size,num_class=len(labels),hidden_size=args.hidden_size)
        # clf = RNN_PretrainedVecs_4(vocab_size=len(vocab),embedding_matrix=embedding_matrix,embed_dim=args.embed_size,num_class=len(labels),hidden_size=args.hidden_size)
    elif model_class=='cnn':
        clf = CNN(vocab_size=len(vocab),embed_dim=args.embed_size,num_class=len(labels))

    if cf=='mlmc':
        loss_func = nn.BCEWithLogitsLoss()
    else:
        loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(clf.parameters(), lr=learning_rate)

    losses = []
    train_losses,train_acc = [],[]
    val_losses, val_acc=[],[]
    min_loss=1000
    total_step=len(train_dataloader)

    for epoch in range(num_epochs):
        train_loss,correct=0,0
        for i,(X,y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            out,y_pred = clf(process_torch_input(X,model_class))
    #         print (y_pred.shape,y.shape)
    #         break
            if (cf=='mlmc'):loss = loss_func(y_pred, y.squeeze(0).float())
            else:loss = loss_func(y_pred, y.squeeze(0))
            train_loss+=loss.item()
            
            if cf=='mc':
                y_pred=torch.argmax(y_pred,dim=1)
                correct+= (y_pred==y).sum().item()
            else:
                y_pred=y_pred>0
                correct+= (y==y_pred.long()).all(dim=1).sum().item()
            
            loss.backward()    
            optimizer.step()
            if (i+1) % 50 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        train_loss=round(train_loss/len(train_dataloader),3)
        acc=round(100*correct/len(train_dataset),3)
        train_losses.append(train_loss)
        train_acc.append(acc)
        print (f"Train loss: {train_loss} Train accuracy: {acc}")  

        val_loss,correct=0,0
        for i,(X,y) in enumerate(val_dataloader):
            out,y_pred=clf(process_torch_input(X,model_class))
            if (cf=='mlmc'):loss = loss_func(y_pred, y.squeeze(0).float())
            else:loss = loss_func(y_pred, y.squeeze(0))
            val_loss+= round(loss.item(),3)

            if cf=='mc':
                y_pred=torch.argmax(y_pred,dim=1)
                correct+= (y_pred==y).sum().item()
            else:
                y_pred=torch.sigmoid(y_pred)>0.5
                correct+= (y==y_pred.long()).all(dim=1).sum().item()
        val_loss=round(val_loss/len(val_dataloader),3)

        if(val_loss<min_loss):
            print ('\nSaving best model...')
            min_loss=val_loss
            torch.save(clf.state_dict(), './best_model.pt')
    
        val_losses.append(val_loss)
        acc=round(100*correct/len(val_dataset),3)
        val_acc.append(acc)
        print (f'Valid loss: {val_loss} Valid acc: {acc}')
        print ('\n'+'='*20)
    

    def pad_sequence(x,longest):
        return x+[0]*(longest-len(x))

    outs,preds=[],[]
    for i,X in enumerate(x_test):
        if(model_class=='cnn'):
            X=pad_sequence(X,max_len)
        out,y_pred=clf(process_torch_input(torch.tensor(X).unsqueeze(0),model_class))
        outs.append(y_pred.detach().numpy())
        # print (out.detach()[0].numpy())
        if cf=='mc':
            if(model_class=='cnn'):
                y_pred=torch.argmax(y_pred)
            else:
                y_pred=torch.argmax(y_pred,dim=1)
            preds.append(labels_mapping_rev[y_pred.item()])
        else:
            y_pred=torch.sigmoid(y_pred)>0.5
            if(model_class=='cnn'):
                preds.append(labels[y_pred.detach().numpy()])
            else:
                preds.append(labels[y_pred.detach().numpy()[0]])
        
    if cf=='mlmc':
        preds_=[]
        for i in preds:
            i=list(i)
            if len(i)==0 or i==['none']:
                preds_.append('none')
            elif 'none' in i and len(i)>1:
                i.remove('none')
                preds_.append(('_').join(sorted(i)))
            else:
                preds_.append(('_').join(sorted(i)))
        preds=preds_
   
    write_to_csv('./outs.csv',preds)

    if args.unsupervised==True:
        print ('Moving to unsupervised predictions...')
        from unsupervised import find_similarity,dependency_parsing
        preds=find_similarity(test_df.utterances.values,preds)

        preds=dependency_parsing(test_df.utterances.values,preds)

    write_to_csv('./outs'+str(model_class)+'.csv',preds)

    outs= np.array(outs)
    print (outs.shape)
    return outs,labels_mapping_rev
    
