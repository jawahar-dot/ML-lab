import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'naivetext.csv')

msg = pd.read_csv(csv_path, names=['message', 'label'])
print('The dimensions of the dataset:', msg.shape)

msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum
print("\nMessages:\n", X.head())
print("\nLabels:\n", y.head())

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nThe total number of Training Data:", ytrain.shape[0])
print('The total number of Test Data:', ytest.shape[0])

count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)
print("\nThe words or Tokens in the text documents:\n")
print(count_vect.get_feature_names_out())

df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names_out())
print("\nSample Bag-of-Words representation:\n", df.head())

clf = MultinomialNB().fit(xtrain_dtm, ytrain)

predicted = clf.predict(xtest_dtm)

print('\nAccuracy of the classifier is:', metrics.accuracy_score(ytest, predicted))
print('\nConfusion matrix:\n', metrics.confusion_matrix(ytest, predicted))
print('\nThe value of Precision:', metrics.precision_score(ytest, predicted))
print('\nThe value of Recall:', metrics.recall_score(ytest, predicted))
