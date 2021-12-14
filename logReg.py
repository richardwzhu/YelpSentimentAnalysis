import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Restaurant_Reviews.txt', delimiter="\t", quoting=3)

corpus = []

import nltk
import re
from stop_list import closed_class_stop_words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from stop_list import closed_class_stop_words

for i in range(0,1000):
    # Cleaning special character from the reviews
    line = re.sub(pattern='[^a-zA-Z]', repl=' ', string=data['Review'][i])

    # Converting the entire review into lower case
    line = line.lower()

    # Tokenizing the review by words
    line_words = line.split()

    # Removing the stop words
    line_words = [word for word in line_words if not word in closed_class_stop_words]

    # Stemming the words
    ps = PorterStemmer()
    line = [ps.stem(word) for word in line_words]

    # Joining the stemmed words
    line = ' '.join(line)
    corpus.append(line)

#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:,-1].values

#splitting into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#logisitic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_model = model.predict(X_test)

#Analysing model
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_model))
#               precision    recall  f1-score   support
#
#            0       0.79      0.82      0.81        97
#            1       0.83      0.80      0.81       103

#     accuracy                           0.81       200
#    macro avg       0.81      0.81      0.81       200
# weighted avg       0.81      0.81      0.81       200