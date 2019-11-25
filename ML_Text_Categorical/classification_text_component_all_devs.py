import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.stem import SnowballStemmer
import logging
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def preprocess_text(text):
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    tokens = re.findall('[a-z]+', text)
    tokens = [token for token in tokens if len(token) > 1 and token not in stop_words]
    combined_text = ' '.join(stemmer.stem(each_token) for each_token in tokens)
    return combined_text

def extract_features(features, texts, train):
    if train:
        vectorizer = DictVectorizer()
        vectorizer.fit(features)
        with open('./trained_models/dict_vectorizer.pickle', 'wb') as file:
            pickle.dump(vectorizer, file)
    else:
        with open('./trained_models/dict_vectorizer.pickle', 'rb') as file:
            vectorizer = pickle.load(file)
    nominal_features = vectorizer.transform(features).toarray()

    if train:
        tf_idf = TfidfVectorizer(analyzer='word', sublinear_tf=True)
        tf_idf.fit(texts)
        with open('./trained_models/tfidf.pickle', 'wb') as file:
            pickle.dump(tf_idf, file)
    else:
        with open('./trained_models/tfidf.pickle', 'rb') as file:
            tf_idf = pickle.load(file)
    text_features = tf_idf.transform(texts).toarray()
    all_features = np.concatenate((nominal_features, text_features), axis=1)
    return all_features

def classify_liblinear_topn(features, true_classes, classes_train, train, topn):
    accuracy = 0
    predicted_classes = []

    if train:
        logistic_regression = LogisticRegression(solver='liblinear', penalty='l2', multi_class='ovr', max_iter=50000, C=5.0)
        logistic_regression.fit(features, true_classes)
        pickle.dump(logistic_regression, open('_liblinear.pickle', 'wb'))
    else:
        logistic_regression = pickle.load(open('_liblinear.pickle', 'rb'))
        predicted_classes = logistic_regression.predict(features)
        probabilities = logistic_regression.predict_proba(features)
        i = 0
        for each_sample in probabilities:
            indices = each_sample.argsort()[-topn:][::-1]
            topn_dev = []
            for k in range(topn):
                topn_dev.append(classes_train[indices[k]])
            if true_classes[i] in topn_dev:
                predicted_classes[i] = true_classes[i]
            i = i+1
        accuracy = accuracy_score(true_classes, predicted_classes)
    return accuracy

def convert_classes(fixers):
    label_encoder = LabelEncoder()
    classes = label_encoder.fit_transform(fixers)
    return classes

br_df = pd.read_csv('./resources/eclipse_bugs_data_new.csv')
br_df = br_df.fillna('')

br_df['summary'] = br_df['summary'].str.lower()
br_df['description'] = br_df['description'].str.lower()
br_df['summary'] = br_df['summary'] + ' ' + br_df['description']
br_df['summary'] = br_df['summary'].apply(preprocess_text)
br_df['created_on'] = pd.to_datetime(br_df.created_on)
br_df['class'] = pd.Series(convert_classes(br_df['fixer_names']))
print(len(np.unique(br_df['class'].values)))
br_df.sort_values('created_on', inplace=True)
br_df.reset_index(inplace=True)
br_df.drop(columns=['index'], inplace=True)

splitted_sets = []
for group, df in br_df.groupby(np.arange(len(br_df)) // 100):
    splitted_sets.append(df)

accuracies = []
precisions = []
recalls = []
predictions = []
top3_accuracies = []
top5_accuracies = []

br_df = splitted_sets[0]
step = 0
for each_set in splitted_sets[1:]:
    br_df_test = each_set

    train_size = len(br_df.index)
    test_size = len(br_df_test.index)

    train_nominal = [br_df.iloc[index][['product', 'component']] for index in range(0, train_size)]
    test_nominal = [br_df_test.iloc[index][['product', 'component']] for index in range(0, test_size)]

    features_train = extract_features(train_nominal, br_df['summary'], True)
    features_test = extract_features(test_nominal, br_df_test['summary'], False)

    classes_train = np.unique(br_df['class'].values)

    classify_liblinear_topn(features_train, br_df['class'].values, classes_train, True, 1)
    accuracy = classify_liblinear_topn(features_test, br_df_test['class'].values, classes_train, False, 1)
    accuracy_top3 = classify_liblinear_topn(features_test, br_df_test['class'].values, classes_train, False, 3)
    accuracy_top5 = classify_liblinear_topn(features_test, br_df_test['class'].values, classes_train, False, 5)

    accuracies.append(accuracy)
    top3_accuracies.append(accuracy_top3)
    top5_accuracies.append(accuracy_top5)

    br_df = br_df.append(each_set, ignore_index=True, sort=False)
    print(str(step)+' Done')
    step = step + 1

print(np.mean(accuracies))
print(np.mean(top3_accuracies))
print(np.mean(top5_accuracies))



