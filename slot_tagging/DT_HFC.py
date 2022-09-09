import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

train_df=pd.read_csv('./train_data.csv')
train_df.head()

test_df=pd.read_csv('./test_data.csv')
test_df.head()

test_tags=[(' ').join(['O']*len(i.split())) for i in test_df['utterances'].values]
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

s_id,prevprev_word,prev_word,word,next_word,nextnext_word,prevprev_tag,prev_tag,tag,count=[],[],[],[],[],[],[],[],[],[]
for idx,row in tqdm(all_df.iterrows()):
    u_words,u_tags= row['utterances'].split(),row['IOB Slot tags'].split()
    assert len(u_words)==len(u_tags)
    for i in range(len(u_words)):
        s_id.append(idx)
        if i==0:
            prevprev_word.append('<START>')
            prev_word.append('<START>')
            # prevprev_tag.append('<START2>')
            # prev_tag.append('<START1>')
        elif i==1:
            prevprev_word.append('<START>')
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
        
            
    
#     break

word_df=pd.DataFrame(zip(s_id,prevprev_word,prev_word,word,next_word,nextnext_word,count,tag),columns=['s_id','prevprev_word','prev_word','word','next_word','nextnext_word','count','tag'])
y=word_df['tag'].values
num_train_words=sum(word_df['s_id']<len(train_df))
num_test_words=len(word_df)-num_train_words
word_df=word_df.drop(columns=['s_id','tag','count'])
# word_df=word_df.drop(columns=['s_id','tag'])
print (word_df.head(10))

print ('num_train_words,num_test_words: ',num_train_words,num_test_words)

from sklearn.feature_extraction import DictVectorizer
dvect = DictVectorizer(sparse=False)
X = dvect.fit_transform(word_df.to_dict("records"))
X.shape,y.shape

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X[:num_train_words],y[:num_train_words], test_size=0.000002, random_state=42)
print ('X_train.shape,X_val.shape: ',X_train.shape,X_val.shape)
print ('y_train.shape,y_val.shape ',y_train.shape,y_val.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

# 7 Define and train model
clf = DecisionTreeClassifier(random_state=243)
# clf = SGDClassifier(random_state=243,learning_rate='adaptive',eta0=0.1)
# clf = SGDClassifier(random_state=243)
clf.fit(X_train,y_train)


yp_train=clf.predict(X_train)
print ('Train accuracy: ',sum([yp_train[i]==y_train[i] for i in range(len(y_train))])/len(y_train))


yp_val=clf.predict(X_val)
print ('Val accuracy: ',sum([yp_val[i]==y_val[i] for i in range(len(y_val))])/len(y_val))


yp_val= clf.predict(X[num_train_words:])
len(yp_val)

preds = pd.DataFrame(zip(range(len(yp_val)),yp_val),columns=['Id','Predicted'])
preds.head()
preds.to_csv('./preds.csv',index=None)