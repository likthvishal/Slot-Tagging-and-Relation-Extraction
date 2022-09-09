# Relation Extraction using semisupervised learning

Run the code using the command:

`python3 main.py`

Use `python3 main.py -h` to see all arguments. The following arguments are available for experiments:

1. `model_name`: nn/ cnn/ rnn
1. `vectorizer`: tfidf/count
1. `unsupervised`: True/False

Examples:

1. `python main.py --classification mc --n_epochs 15 --lr 0.001 --bs 32 --threshold 10`
1. `python fusion_main.py`
1. `python main.py --classification mc --n_epochs 15 --lr 0.001 --bs 32 --model_class rnn`
1. `python main.py --classification mc --n_epochs 25 --lr 0.001 --vectorizer tfidf --bs 8`

The following features are implemented in the code:

1. Score fusion
1. NER tagging 
1. POS tagging
1. Oversampling
1. Undersampling
1. Out of domain data addition

All parameters will be tuned via grid-search. 

Dependencies:

1. spacy==3.1.3
1. scikit-learn==0.24.2
1. scipy==1.7.0
1. pandas==1.3.2
1. numpy==1.19.5
