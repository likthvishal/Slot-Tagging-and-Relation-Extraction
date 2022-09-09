
# random.seed(1)
import argparse

import warnings
warnings.filterwarnings("ignore")

from helpers import *
from pipeline import *
from classify import *
seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def init_argparse():
    arg_parser = argparse.ArgumentParser()

    # arg_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--classification', default='mc', type=str, help='multilabel multiclass (mlmc) or multiclass (mc)')
    arg_parser.add_argument('--model_class', default='nn', type=str, help='nn/rnn/lstm/cnn')
    arg_parser.add_argument('--train_file', default='./train_data_merged_labels.csv', type=str, help='train csv file')
    arg_parser.add_argument('--test_file', default='./test_data.csv', type=str, help='test csv file')

    arg_parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    arg_parser.add_argument('--bs', default=16, type=int, help='batch size')
    arg_parser.add_argument('--n_epochs', default=20, type=int, help='num epochs')
    arg_parser.add_argument('--test_split', default=0.2, type=float, help='test split')
    
    arg_parser.add_argument('--threshold', default=10, type=int, help='remove labels less than threshold samples')
    arg_parser.add_argument('--embed_size', default=500, type=int, help='embedding size')
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='hidden size')

    arg_parser.add_argument('--vectorizer', default='count', type=str, help='tfidf or count or count_tfidf')
    arg_parser.add_argument('--oversample', default=False, type=bool, help='oversample')
    arg_parser.add_argument('--undersample', default=False, type=bool, help='undersample')
    arg_parser.add_argument('--ner', default=False, type=bool, help='append ner tags to sentence')
    arg_parser.add_argument('--pos', default=False, type=bool, help='append pos tags to sentence')
    arg_parser.add_argument('--out_of_domain', default=False, type=bool, help='add more out-of-domain gender data ')
    arg_parser.add_argument('--unsupervised', default=False, type=bool, help='add more out-of-domain gender data ')
    return (arg_parser.parse_args())

def main():
    args=init_argparse()
    classify(args)

main()
