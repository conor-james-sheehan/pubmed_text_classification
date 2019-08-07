import pandas as pd
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

VOCAB = [
    # background
    'previous', 'studies', 'known', 'recently',
    # objective
    'aim', 'aimed', 'objective', 'goal', 'intention', 'purpose', 'target',
    # method
    'using', 'used', 'use', 'method', 'experiment', 'by'
    # result
    'found', 'find', 'finding', 'detect', 'detected', 'detecting' 'measure', 'measured', 'measuring', 'measurement',
    'reveal', 'revealed', 'revealing', 'identify', 'identified', 'identifying', 'identification', 'indicate',
    'indicated', 'indicating', 'result',
    # conclusion
    'conclusion', 'conclude', 'summary', 'summarise', 'summarize', 'valuable', 'important', 'demonstrate',
    'demonstrated', 'demonstrating', 'new', 'novel'
]


def main():
    os.chdir('..')
    data_dir = os.path.join('datasets', 'pubmed20k')
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'), index_col=0)
    X_train = train['sentence'].values
    X_test = test['sentence'].values
    y_train = train['label'].values
    y_test = test['label'].values

    count_vectorizer = CountVectorizer(vocabulary=None)
    # clf = RandomForestClassifier(min_samples_leaf=100, n_estimators=10)
    clf = LogisticRegressionCV()
    model = Pipeline([
        ('count_vectorizer', count_vectorizer),
        ('clf', clf)
    ])

    # param_grid = {'clf__min_samples_leaf': [100, 50, 25],
    #               'clf__n_estimators': [10, 100, 200]}

    # model = GridSearchCV(model, param_grid=param_grid, n_jobs=-2)

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='micro')
    best_words = pd.Series(dict(zip(count_vectorizer.vocabulary_, clf.feature_importances_))).sort_values(ascending=False)
    cm = confusion_matrix(y_test, y_pred_test)
    with open(os.path.join(data_dir, 'labels_mapping.json'), 'r') as infile:
        label_strings = json.load(infile)
    cm = pd.DataFrame(cm, columns=label_strings, index=label_strings)
    print('acc: {:.3f}\tf1: {:.3f}'.format(accuracy, f1))


if __name__ == '__main__':
    main()
