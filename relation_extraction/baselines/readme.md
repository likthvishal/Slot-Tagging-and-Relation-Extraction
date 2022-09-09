# Relation Extraction using semisupervised learning

Run the code using the command:

`python3 main.py`

Use `python3 main.py -h` to see all arguments. The following arguments are available for experiments:

1. `model_name`: SGD/ RandomForest / DecisionTree/ NaiveBayes/ SVC
1. `vectorizer`: tfidf/count
1. `oversample/undersample/ner/pos/out_of_domain/unsupervised`: True/False


The following features are implemented in the code:

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
