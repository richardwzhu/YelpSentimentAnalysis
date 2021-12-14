import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from contractions import contractions

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

file = open("RestaurantReviews.txt", "r")
data = file.readlines()

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

df = pd.DataFrame(columns=('text', 'label'))

for i in range(len(data)):
    review = data[i].split("\t")
    if len(review) != 2:
        continue
    text = review[0]
    label = review[1] 
    df.loc[i] = [text, label]

#Preprocessing: lower case, contractions, non-alpha characters, stop words, lemmatizer
df['preprocess']=df['text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
df['preprocess']=df['preprocess'].apply(lambda x:contractions(x))
df['preprocess']=df['preprocess'].apply(lambda x: " ".join([re.sub('[^A-Za-z]+','', x) for x in nltk.word_tokenize(x)]))
df['preprocess']=df['preprocess'].apply(lambda x: " ".join([x for x in x.split() if x not in stop]))
df['preprocess']=df['preprocess'].apply(lambda x: " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]))

#Splitting into training, development, and testing sets
df = df.sample(frac = 1)
X_train = df['preprocess'][:800]
X_dev = df['preprocess'][800:900]
X_test = df['preprocess'][900:]
Y_train = df['label'][:800]
Y_dev = df['label'][800:900]
Y_test = df['label'][900:]
    
#Unigram TFIDF
unigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
unigram_x_train = unigram_tfidf_vectorizer.fit_transform(X_train)
unigram_x_dev = unigram_tfidf_vectorizer.transform(X_dev)

#Training/Development: fit training data, predict test data, analyze results
clf = LinearSVC(random_state=0)
clf.fit(unigram_x_train, Y_train)
y_dev_pred=clf.predict(unigram_x_dev)
report=classification_report(Y_dev, y_dev_pred)

#Testing
unigram_x_test = unigram_tfidf_vectorizer.transform(X_test)
y_test_pred=clf.predict(unigram_x_test)
report=classification_report(Y_test, y_test_pred)
print(report)
#               precision    recall  f1-score   support
#            0       0.86      0.85      0.85        52
#            1       0.84      0.86      0.85        49
#     accuracy                           0.85       101
#    macro avg       0.85      0.85      0.85       101
# weighted avg       0.85      0.85      0.85       101

#Bigram TFIDF
bigram_tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,2))
bigram_x_train = bigram_tfidf_vectorizer.fit_transform(X_train)
bigram_x_dev = bigram_tfidf_vectorizer.transform(X_dev)

clf.fit(bigram_x_train, Y_train)
y_dev_pred=clf.predict(bigram_x_dev)
report=classification_report(Y_dev, y_dev_pred)

bigram_x_test = bigram_tfidf_vectorizer.transform(X_test)
y_test_pred=clf.predict(bigram_x_test)
report=classification_report(Y_test, y_test_pred)
print(report)
#               precision    recall  f1-score   support
#            0       0.54      0.92      0.68        52
#            1       0.67      0.16      0.26        49
#     accuracy                           0.55       101
#    macro avg       0.60      0.54      0.47       101
# weighted avg       0.60      0.55      0.48       101

#Unigram Counts
unigram_count_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_count_x_train = unigram_count_vectorizer.fit_transform(X_train)
unigram_count_x_dev = unigram_count_vectorizer.transform(X_dev)

clf.fit(unigram_count_x_train, Y_train)
y_dev_pred=clf.predict(unigram_count_x_dev)
report=classification_report(Y_dev, y_dev_pred)

unigram_count_x_test = unigram_count_vectorizer.transform(X_test)
y_test_pred=clf.predict(unigram_count_x_test)
report=classification_report(Y_test, y_test_pred)
print(report)
#               precision    recall  f1-score   support
#           0        0.82      0.87      0.84        52
#           1        0.85      0.80      0.82        49
#     accuracy                           0.83       101
#    macro avg       0.83      0.83      0.83       101
# weighted avg       0.83      0.83      0.83       101