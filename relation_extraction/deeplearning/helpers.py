import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import random
import numpy as np
seed=1
np.random.seed(seed)
random.seed(seed)

def build_embmat(model,vocab):
    size=model.vector_size
    model_vocab=set(model.index_to_key)
    vectors=[]
    count=0
    for i in vocab:
        if i in model_vocab:
            vectors.append(model[i])
        else:
            vectors.append(np.zeros(size,dtype='float'))
            count+=1
    vectors=np.array(vectors)
    print (f'{count} words not in the pretrained embeddings!!!!!')
    return vectors

def remove_lowfreq_class(df,threshold=20):
    '''
    threshold: Anything that occurs less than this will be removed.

    '''
    value_counts = df['Core Relations'].value_counts() # Entire DataFrame 
    to_remove = value_counts[value_counts <= threshold].index
    df=df[~df['Core Relations'].isin(to_remove)]
    return df

['actor','country','character','director','producer','genre']
def combine_objs(x,objs):
    y=x
    x=x.split()
    combs=[x[i:j] for i in range(len(x)) for j in range(i+1,len(x)+1)]
    combs=sorted(combs,key=len,reverse=True)
    used={i:False for i in x}
    
    def can_use(words_set):
        for i in words_set:
            if(used[i]==True):
                return False
        return True
    
    def using(words_set):
        for i in words_set:
            used[i]=True
            
    for words_set in combs:
        sent=(' ').join(words_set)
        if sent in objs.keys() and can_use(words_set):
            using(words_set)
            if(objs[sent] in ['actor','character','director','producer','subject','genre','country']):
                y=y.replace(sent,'<'+objs[sent]+'> '+sent)
    return y
    
# def combine_objs(x,label_file,tag):
#     '''
#     input: get german films
#     output: get german language films
#     '''

#     objs=read_file(label_file)

#     y,x=x,x.split()
#     combs=[x[i:j] for i in range(len(x)) for j in range(i+1,len(x)+1)]
#     combs=sorted(combs,key=len,reverse=True)
#     for words_set in combs:
#         sent=(' ').join(words_set)
#         if sent in objs:
#                 y=y.replace(sent,f'{sent} {tag}')
#     return y


def oversample(x,y):
    ros = RandomOverSampler(random_state=1,sampling_strategy='not majority')
    x, y = ros.fit_resample(x, y)
    return x,y


def undersample(x,y):
    ros = RandomUnderSampler(random_state=1)
    x, y = ros.fit_resample(x, y)
    return x,y

def out_of_domain(tag):
    f= read_file('./extra_data/gentleman.txt')
    print(len(f))
    f+= read_file('./extra_data/lady.txt')
    print (f'Adding {len(f)} lines')
    df=pd.DataFrame(zip(f,[tag]*len(f)),columns=['utterances','Core Relations'])
    return df


def read_file(f):
    l=open(f).readlines()
    l=[i[:-1] for i in f]
    return l

def write_to_file(filename,l):
    with open(filename,'w') as f:
        for i in l:
            f.write(str(i)+'\n')

def write_to_csv(filename,predictions):
    ids=list(range(len(predictions)))
    out_df=pd.DataFrame(zip(ids,predictions),columns=['Id','Predicted'])
    out_df.to_csv(filename,index=None)


def process_all_data(args,train_df,test_df):
    print ('Threshold',args.threshold)
    if args.threshold!=0:
        train_df=remove_lowfreq_class(train_df,threshold=args.threshold)

    if args.out_of_domain==True:
        print (f'Before adding out of domain data {len(train_df)}')
        gender_df=out_of_domain('actor.gender')
        train_df=pd.concat([train_df,gender_df])
        print (f'After adding out of domain data {len(train_df)}')
        
    if args.ner==True:
        from unsupervised import get_ner
        print ('Addings NER tags to all sentences..')
        train_df['utterances']=train_df['utterances'].apply(get_ner)
        test_df['utterances']=test_df['utterances'].apply(get_ner)
    
    if args.pos==True:
        print ('Adding POS tags to all sentences..')
        from unsupervised import get_postag
        train_df['utterances']=train_df['utterances'].apply(get_postag)
        test_df['utterances']=test_df['utterances'].apply(get_postag)
    
    return train_df,test_df


def sampling(args,x_train,y_train):

    if args.oversample==True:
        print ("Distribution of classes before oversampling..",np.unique(y_train,return_counts=True))
        x_train,y_train=oversample(x_train,y_train)
        print ("Distribution of classes after oversampling..",np.unique(y_train,return_counts=True))

    if args.undersample==True:
        print ("Distribution of classes before undersampling..",np.unique(y_train,return_counts=True))
        x_train,y_train=undersample(x_train,y_train)
        print ("Distribution of classes after undersampling..",np.unique(y_train,return_counts=True))
    
    return x_train,y_train