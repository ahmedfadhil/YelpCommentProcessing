import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

yelp = pd.read_csv('yelp.csv')
yelp.head()
yelp.info()
yelp.describe()
yelp['text length'] = yelp['text'].apply(len)

g = sns.FacetGrid(yelp, col='stars')
g.map(plt.hist, 'text length', bins=50)

sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')

sns.countplot(x='stars', data=yelp, palette='rainbow')

stars = yelp.groupby('stars').mean()

stars.corr()

sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

X = yelp['text']
y = yelp['stars']

cv = CountVectorizer()

X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

nb = MultinomialNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

pipeline = Pipeline([('bow', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', MultinomialNB())])

x = yelp['text']
y = yelp['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


