
# random.seed(1)
import argparse

import warnings
warnings.filterwarnings("ignore")

from helpers import *
from pipeline import *
from fuse import *
seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import scipy

class argparse_custom:
    def __init__(self,**kwargs):
        for attr,value in kwargs.items():
            setattr(self,attr,value)

def main():
    args_={'bs':32, 'classification':'mc', 'embed_size':500, 'hidden_size':256, 'lr':0.001, 'model_class':'nn', 'n_epochs':15, 'ner':False, 'out_of_domain':False, 'oversample':False, 'pos':False, 'test_file':'./test_data.csv', 'test_split':0.01, 'threshold':10, 'train_file':'./train_data_merged_labels.csv', 'undersample':False, 'unsupervised':False, 'vectorizer':'count', 'pretrained':False}
    args=argparse_custom(**args_)
    model1,labels_mapping_rev=classify_fc(args)
    model1=model1.squeeze(1)

    args={'bs':32, 'classification':'mc', 'embed_size':300, 'hidden_size':256, 'lr':0.001, 'model_class':'rnn', 'n_epochs':15, 'ner':False, 'out_of_domain':False, 'oversample':False, 'pos':False, 'test_file':'./test_data.csv', 'test_split':0.01, 'threshold':10, 'train_file':'./train_data_merged_labels.csv', 'undersample':False, 'unsupervised':False, 'vectorizer':'count','pretrained':False}
    args=argparse_custom(**args)
    model2,labels_mapping_rev=classify_fc(args)
    model2=model2.squeeze(1)


    model1=scipy.special.softmax(model1, axis=1)
    model2=scipy.special.softmax(model2, axis=1)
    print ('m1:',model1.shape)
    print ('m2:',model2.shape)
    
    fused_emb=np.sum([model1,model2],axis=0)
    print (fused_emb.shape)

    
    preds=[labels_mapping_rev[i] for i in np.argmax(model1,axis=1)]
    write_to_csv('./outs.csv',preds)

    preds=[labels_mapping_rev[i] for i in np.argmax(model2,axis=1)]
    write_to_csv('./outs.csv',preds)

    preds=[labels_mapping_rev[i] for i in np.argmax(fused_emb,axis=1)]
    write_to_csv('./outs.csv',preds)


main()
