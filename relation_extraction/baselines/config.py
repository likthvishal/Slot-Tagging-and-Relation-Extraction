
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

classes={
    'SGD': SGDClassifier(),
    'RandomForest': RandomForestClassifier(),
    'DecisionTree':DecisionTreeClassifier(),
    'NaiveBayes':MultinomialNB(),
    'SVC':SVC(),
}

# Choose some parameter combinations to try
parameters = { 'SGD':
                    {'loss': ['hinge','log'], 
                    'penalty': ['l2', 'l1'], 
                    'alpha': [0.0001,0.01],
                    'max_iter': [ 5, 15,45], 
                    'random_state':[1]
                    },

                'RandomForest':
                    {'n_estimators': [10, 50], 
                    'max_features': ['log2', 'sqrt'], 
                    'criterion': ['entropy', 'gini'],
                    'max_depth': [ 5, 10], 
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1,10]
                    },

                'SVC':
                    {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 
                    'class_weight': ['None', 'balanced'], 
                    'degree': [2,5,10],
                    'random_state': [1], 
                    },

                'DecisionTree':
                    {
                    'class_weight': ['None', 'balanced'], 
                    'criterion':['gini', 'entropy'],
                    'random_state': [1], 
                    'min_samples_split':[2,5,10]
                    },

                'NaiveBayes':
                    {'alpha':[0.2,0.5,1.0],
                    'fit_prior':[True,False]
                    }

}

