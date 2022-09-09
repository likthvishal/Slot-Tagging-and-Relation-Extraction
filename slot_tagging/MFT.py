import os
import nltk
from sklearn import metrics
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(1)

def get_accuracy(test_sentences, tagged_test_sentences):
    gold = [str(tag) for sentence in test_sentences for token, tag in sentence]
    pred = [str(tag) for sentence in tagged_test_sentences for token, tag in sentence]
    return(metrics.accuracy_score(gold, pred))


train_df=pd.read_csv('./train_data.csv')
train_df.head()

test_df=pd.read_csv('./test_data.csv')
test_df.head()

test_tags=[(' ').join(['O']*len(i.split())) for i in test_df['utterances'].values]
print (len(test_tags))
test_df['IOB Slot tags']=test_tags
test_df.head()

def join_words(df):
    vocab={}
    sents=df.values
    data=[[],[]]
    for pair in sents:
        sent,tags=pair[0].split(),pair[1].split()
        phrase=[]
        word_phrase=[]
        prev=None
        for j in range(len(sent)):
            cur_tag=tags[j]
            if(prev=='O' or prev==None or cur_tag=='O'):
                phrase.append(cur_tag)
                word_phrase.append(sent[j])
            else:
                phrase[-1]+='|'+cur_tag
                word_phrase[-1]+='|'+sent[j]
            prev=cur_tag
        for i in word_phrase:
            if('|' in i):
                vocab[i.replace('|',' ')]=i
        data[0].append((' ').join(word_phrase))
        data[1].append((' ').join(phrase))

    return pd.DataFrame(zip(data[0],data[1])),vocab


train_df,vocab=join_words(train_df)
train_df.head()

new_vocab = {}
for k in sorted(vocab, key=lambda x:len(x.split()), reverse=True):
    new_vocab[k] = vocab[k]
vocab=new_vocab
del vocab['e t']

def join_test_words(vocab,df):
    new=[]
    sents=df['utterances'].values
    for k,v in vocab.items():
        for s in sents:
            if((set(k.split()).issubset(set(s.split()))) and (k+'|' not in s) and ('|'+k not in s)):
                # print (k,',',s)
                s=s.replace(k,v)
                # print (s)
            # print (k,s)
            new.append(s)
        sents=new
        new=[]
    return pd.DataFrame(zip(sents,df['IOB Slot tags'].values.tolist()))


test_df_=join_test_words(vocab,test_df)
test_df_.head()
test_df=test_df_

def df2tuple(df):
    tuples=[]
    for i in df.values:
        tuples.append(list(zip(i[0].split(),i[1].split())))
    return tuples

train_sentences=df2tuple(train_df)
val_length=int(0.8*len(train_sentences))
val_sentences=df2tuple(train_df[-val_length:])

test_sentences=df2tuple(test_df)
print (len(train_sentences),len(test_sentences),len(val_sentences))


def baseline_trainer(sentences):
    pair_counts=Counter([pair for sent in sentences for pair in sent])
    word_val=defaultdict(int)
    word_tag={}
    top_tag=defaultdict(int)
    for pair,val in pair_counts.items():
        word,tag=pair[0],pair[1]
        top_tag[tag]+=val
        if(word_val[word]<val):
            word_val[word]=val
            word_tag[word]=tag
    top_tag=dict(top_tag)
    return word_tag,max(top_tag,key=top_tag.get)

def baseline_tagger(sentences,word_tag,top_tag):
    predicted=[]
    for sent in sentences:
        untagged=[]
        for pair in sent:
            try:untagged.append((pair[0],word_tag[pair[0]]))
            except:untagged.append((pair[0],top_tag))
        predicted.append(untagged)
    return predicted

print ('\nPreparing baseline tagger....\n')
baseline_wordtag,baseline_toptag=baseline_trainer(train_sentences)
tagged_test_sentences=baseline_tagger(test_sentences,baseline_wordtag,baseline_toptag)

print ('\Train baseline accuracy....')
tagged_train_sentences=baseline_tagger(train_sentences,baseline_wordtag,baseline_toptag)
print (get_accuracy(tagged_test_sentences=tagged_train_sentences,test_sentences=train_sentences))

print ('\nValidation baseline accuracy....')
tagged_val_sentences=baseline_tagger(val_sentences,baseline_wordtag,baseline_toptag)
print (get_accuracy(tagged_test_sentences=tagged_val_sentences,test_sentences=val_sentences))

tagged_=[]
for sent in tagged_test_sentences:
    new=[]
    for phrase in sent:
        wrds=phrase[0].split('|')
        tgs=phrase[1].split('|')            
        new+=[tuple((i,j)) for i,j in list(zip(wrds,tgs))]
    tagged_.append(new)

words=[j for i in tagged_ for j in i]
tags=[j[1] for i in tagged_ for j in i]
words=[i for i in range(len(words))]

preds = pd.DataFrame(zip(words,tags),columns=['Id','Predicted'])
preds.head()
assert len(preds)==6415
preds.to_csv('./preds.csv',index=None)