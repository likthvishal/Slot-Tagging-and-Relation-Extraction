import pandas as pd
import numpy as np

test = pd.read_csv('hw1_test.csv')
test = test['utterances'].apply(lambda x:x.split()).tolist()
print (test[0])

df = pd.read_csv('preds/bert_cr.csv')
crs = df['Predicted'].astype('str').tolist()

df = pd.read_csv('preds/bert_st.csv')
sts = df['Predicted'].tolist()

a=0
with open('prediction.txt','w') as f:
    for i in range(len(test)):
        if crs[i]=='none' or crs[i]=='nan':
            cr = ''
        else: cr = crs[i]
        st = (' ').join(sts[a:a+len(test[i])])
        # print (cr, st, type(cr), type(st))
        f.write(st+'\t'+cr+'\n')
        a+=len(test[i])

