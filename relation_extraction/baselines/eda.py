
def get_distrib(train_df,test_df):
    tr_wrds,te_wrds=[],[]
    for i in train_df.utterances.values:
        tr_wrds+=i.split()
    for i in test_df.utterances.values:
        te_wrds+=i.split()
        
    print ('Number of words in train',len(set(tr_wrds)))
    print ('Number of words in test',len(set(te_wrds)))
    print (f'{len(set(te_wrds)-set(tr_wrds))} words in test not present in train')
