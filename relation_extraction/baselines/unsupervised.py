from scipy.spatial.distance import cosine
from helpers import write_to_csv
import numpy as np
import random
random.seed(1)

import spacy
print('Loading spacy model (might take some time)...')
nlp=spacy.load('en_core_web_sm')
print ('Done loading')

def load_glove():
    print ('loading glove vectors..')
    embeddings_index = dict()
    f = open('./glove.6B.200d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    glove_vocab=set(embeddings_index.keys())

    return embeddings_index,glove_vocab

def get_postag(x):
    doc=nlp(x)
    pos=[]
    for token in doc:
        pos.append(token.pos_)
    return x+' '+(' '.join(pos))


def get_ner(x):
    doc=nlp(x)
    ents=[]
    for ent in doc.ents:
        ents.append(ent.label_)
    return x+' '+(' '.join(ents))

def get_gmean(sent,embeddings_index,glove_vocab):
    #mean of words vectors in a sentence
    return np.mean([embeddings_index[i] for i in sent.split() if i in glove_vocab],axis=0)

# get_gmean('which female actors starred in hitch')


def get_ncs(sents):
    #obtain noun chunks from a list of sentences
    ncs=[]
    for sent in sents:
        doc=nlp(sent)
        cur_ncs=[]
        for ent in doc.noun_chunks:
            cur_ncs.append(ent.text)
        ncs.append(cur_ncs)
    return ncs


def find_similarity(sents,preds):
    embeddings_index,glove_vocab=load_glove()
    ncs=get_ncs(sents) #get noun chunks

    emb_gender=embeddings_index['gender']
    emb_actor=embeddings_index['actor']

    #calculate similarity with both actor and gender
    gend_ncs_sims=[[cosine(get_gmean(j,embeddings_index,glove_vocab),emb_gender) for j in i] for i in ncs]
    act_ncs_sims=[[cosine(get_gmean(j,embeddings_index,glove_vocab),emb_actor) for j in i] for i in ncs]

    print ('These sentences have label gender...')
    print ('-'*20)
    for i in range(len(gend_ncs_sims)):
        for j in range(len(gend_ncs_sims[i])):
            if (0.1<gend_ncs_sims[i][j]<0.65 and 0.1<act_ncs_sims[i][j]<0.65):
                print (sents[i])
                preds[i]='actor.gender'
    print ('-'*50)
    
    write_to_csv('./outs.csv',preds)
    return preds



def get_ents(sents,indexes=[]):
    print ('Finding entities in utterances..')
    for idx,s in enumerate(sents):
        doc=nlp(s)
        if sum([dp_play(sent.root) for sent in doc.sents])>0:
            indexes.append(idx)
    return indexes

def get_leaf(node):
    if([i for i in node.children]==[]):
        return 1
    return sum([get_leaf(c) for c in node.children])

def dp_play(node,flag=False):
    if node.text.startswith('play'):
        if node.n_lefts+node.n_rights>2 or get_leaf(node)>2:
            return True
    if node.n_lefts + node.n_rights > 0:
        for child in node.children:
            if(dp_play(child,flag)):
                return True
    return flag

def dependency_parsing(sents,preds):
    indexes=get_ents(sents)
    print ('Following sentences have starring character and actor')
    print ('-'*20)

    for i in indexes:
        if(preds[i]=='actor.gender'):
            preds[i]='actor.gender_movie.starring.character'
            continue
        preds[i]='movie.starring.actor_movie.starring.character'
        print (sents[i])
    print ('-'*50)
    
    write_to_csv('./outs.csv',preds)
    return preds
    

