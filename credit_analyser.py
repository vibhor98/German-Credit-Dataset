from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from csv import reader
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
'''n_estimators = 10
max_depth = 10

def load_csv():
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
def clean_dataset():

def prediction(train_features, train_labels, test_features, test_lables):
    clf = RandomForestClassifier(n_estimators, max_depth)
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features, test_lables)

def accuracy(pred, test_labels):
    print 'Accuracy', accuracy_score(pred, test_lables)'''
    
data = pd.read_csv('../Documents/german_credit_data.csv')
print data.head(5)
#no_males = data[ data['Sex'] == 'male']['Credit amount']
#no_females = data[ data['Sex'] == 'female'].shape[0]
#print no_males, no_females
t_features = []
for amt, age, time, h in zip(list(data['Credit amount']), list(data['Age']), list(data['Duration']), list(data['Housing'])):
    t_features.append([amt, age, time, h])

t_labels = list(data['Job'])
train_features = t_features[:900]
test_features = t_features[900:]
train_labels = t_labels[:900]
test_labels = t_labels[900:]
#clf = DecisionTreeClassifier()
#clf = GaussianNB()
#clf = AdaBoostClassifier(n_estimators=100)
clf = RandomForestClassifier(n_estimators=200, max_depth=100)
clf.fit(train_features, train_labels)
pred = clf.predict(test_features)
print pred
print 'Accuracy:', accuracy_score(test_labels,pred)
#print len(train_features), len(train_labels)

######## Data visualisation
plt.plot( test_labels, '.')
plt.ylabel('Type of job')   # Accurate values
plt.title('Prediction of jobs in which credit seekers are engaged')

plt.plot( pred, '.')      # Prediction
plt.show()
