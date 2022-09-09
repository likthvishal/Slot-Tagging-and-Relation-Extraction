from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,make_scorer


import pandas as pd
import numpy as np
from collections import Counter
import random
random.seed(1)
import argparse

import warnings
warnings.filterwarnings("ignore")

from config import parameters,classes
from helpers import *
from pipeline import *
from eda import *

def init_argparse():
    arg_parser = argparse.ArgumentParser()

    # arg_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--train_file', default='./train_data_merged_labels.csv', type=str, help='train csv file')
    arg_parser.add_argument('--test_file', default='./test_data.csv', type=str, help='test csv file')
    arg_parser.add_argument('--model_name', default='SGD', type=str, help='SGD/ RandomForest / DecisionTree/ NaiveBayes/ SVC')
    arg_parser.add_argument('--vectorizer', default='count', type=str, help='tfidf or count')
    arg_parser.add_argument('--oversample', default=False, type=bool, help='oversample')
    arg_parser.add_argument('--undersample', default=False, type=bool, help='undersample')
    arg_parser.add_argument('--ner', default=False, type=bool, help='append ner tags to sentence')
    arg_parser.add_argument('--pos', default=False, type=bool, help='append pos tags to sentence')
    arg_parser.add_argument('--out_of_domain', default=False, type=bool, help='add more out-of-domain gender data ')
    arg_parser.add_argument('--unsupervised', default=False, type=bool, help='add more out-of-domain gender data ')
    return (arg_parser.parse_args())


def main():
    args=init_argparse()

    train_df,test_df=read_data(args.train_file,args.test_file)
    
    # get_distrib(train_df,test_df) 

    train_df,test_df=clean_data(train_df,test_df) 

    train_df=remove_lowfreq_class(train_df,threshold=20)

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
        print ('Addings POS tags to all sentences..')
        from unsupervised import get_postag
        train_df['utterances']=train_df['utterances'].apply(get_postag)
        test_df['utterances']=test_df['utterances'].apply(get_postag)

    print ('Sample Data...')
    print ('-'*20)
    print (train_df.utterances.values[:3])
    print ('-'*20)

    x_train, x_val, y_train, y_val = train_test_split(train_df['utterances'].values, train_df['Core Relations'].values, test_size=0.2, random_state=1, 
                                                    shuffle=True,stratify=train_df['Core Relations'])
    print(x_train.shape,x_val.shape)
    assert (x_train.shape==y_train.shape)
    assert(x_val.shape==y_val.shape)

    x_test=train_df.utterances.values
    x_train,x_val,x_test=vectorize(args.vectorizer,x_train,[x_train,x_val,x_test])
    
    if args.oversample==True:
        print ("Distribution of classes before oversampling..",np.unique(y_train,return_counts=True))
        x_train,y_train=oversample(x_train,y_train)
        print ("Distribution of classes after oversampling..",np.unique(y_train,return_counts=True))

    if args.undersample==True:
        print ("Distribution of classes before undersampling..",np.unique(y_train,return_counts=True))
        x_train,y_train=undersample(x_train,y_train)
        print ("Distribution of classes after undersampling..",np.unique(y_train,return_counts=True))
    

    print ('Train,Val,Test shape',x_train.shape,x_val.shape,x_test.shape)

    model_name=args.model_name


    model_parameters=parameters[model_name]
    clf=classes[model_name]

    # # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # # Run the grid search
    grid_obj = GridSearchCV(clf, model_parameters, scoring=acc_scorer,return_train_score=True)
    grid_obj = grid_obj.fit(x_train, y_train)

    # # Set the clf to the best combination of parameters
    clf = grid_obj.best_estimator_

    # # Fit the best algorithm to the data. 
    clf.fit(x_train, y_train)


    print ('Train Accuracy:', accuracy_score(y_train,clf.predict(x_train)))

    # predict
    predictions = clf.predict(x_val)
    print('Validation Accuracy: ', accuracy_score(y_val,predictions))


    #cross validation
    scores=cross_val_score(clf, x_train, y_train, cv=5)
    print ('Cross validation score: ',sum(scores)/len(scores))
    
    print('Best parameters: ',grid_obj.best_params_)

    predictions=clf.predict(x_test)
    write_to_csv('./outs.csv',predictions)

    if args.unsupervised==True:
        print ('Moving to unsupervised predictions...')
        from unsupervised import find_similarity,dependency_parsing
        predictions=find_similarity(test_df.utterances.values,predictions)

        predictions=dependency_parsing(test_df.utterances.values,predictions)


main()