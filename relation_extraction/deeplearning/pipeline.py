
import pandas as pd
from helpers import combine_objs
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import random
random.seed(1)

def read_data(train,test):
    print (train,test)
    print ('-'*20)
    train_df=pd.read_csv(train)
    print (f'train file has {len(train_df)} samples')
    print (train_df.head(2))
    print ('-'*20)

    test_df=pd.read_csv(test)
    print (f'test file has {len(test_df)} samples')
    print (test_df.head(2))
    print ('-'*20)

    return train_df,test_df

def clean_data(*argv): 
    for df in argv:
        df['utterances']=df['utterances'].apply(lambda x:x.replace('-',''))
        # # df['utterances']=df['utterances'].apply(lambda x: combine_objs(x,'extra_data/languages.txt','language'))
        import json
        with open('./obj_map.json','r') as f:
            objs = json.load(f)
        df['utterances']=df['utterances'].apply(lambda x:combine_objs(x,objs))
    return argv


def vectorize(vtype,train,*argv):
    if vtype=='tfidf':
        vec=TfidfVectorizer()
        vec.fit(train)
    elif vtype=='count':
        vec=CountVectorizer()
        vec.fit(train)
    elif vtype=='count_tfidf':
        vec=TfidfTransformer()
        vec.fit(train)
    transformed_vectors=[]
    for i in range(len(argv[0])):
        transformed_vectors.append(vec.transform(argv[0][i]).todense().A)

    return transformed_vectors
