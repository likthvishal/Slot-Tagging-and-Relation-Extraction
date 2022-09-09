import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import random
random.seed(1)

def remove_lowfreq_class(df,threshold=20):
    '''
    threshold: Anything that occurs less than this will be removed.

    '''
    value_counts = df['Core Relations'].value_counts() # Entire DataFrame 
    to_remove = value_counts[value_counts <= threshold].index
    df=df[~df['Core Relations'].isin(to_remove)]
    return df


def combine_objs(x,label_file,tag):
    '''
    input: get german films
    output: get german language films
    '''

    objs=read_file(label_file)

    y,x=x,x.split()
    combs=[x[i:j] for i in range(len(x)) for j in range(i+1,len(x)+1)]
    combs=sorted(combs,key=len,reverse=True)
    for words_set in combs:
        sent=(' ').join(words_set)
        if sent in objs:
                y=y.replace(sent,f'{sent} {tag}')
    return y


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



