import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.stem import SnowballStemmer
import logging
from _datetime import datetime
import dateutil.relativedelta
import math
import operator
import json

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

def classify_liblinear(features, true_classes, train):
    accuracy = 0

    if train:
        logistic_regression = LogisticRegression(solver='liblinear', penalty='l2', multi_class='ovr', max_iter=50000, C=config["C_parameter"])
        logistic_regression.fit(features, true_classes)
        pickle.dump(logistic_regression, open('./trained_models/liblinear.pickle', 'wb'))
    else:
        logistic_regression = pickle.load(open('./trained_models/liblinear.pickle', 'rb'))
        predicted_classes = logistic_regression.predict(features)
        accuracy = accuracy_score(true_classes, predicted_classes)

    return accuracy

def classify_liblinear_topn(features, true_classes, classes_train, train, topn):
    accuracy = 0
    predicted_classes = []

    if train:
        logistic_regression = LogisticRegression(solver='liblinear', penalty='l2', multi_class='ovr', max_iter=50000, C=config["C_parameter"])
        logistic_regression.fit(features, true_classes)
        pickle.dump(logistic_regression, open('./trained_models/liblinear.pickle', 'wb'))
    else:
        logistic_regression = pickle.load(open('./trained_models/liblinear.pickle', 'rb'))
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

def remove_less_active_devs(br_df):
    active_dev_set = set()
    for comp in np.unique(br_df['component'].values):
        br_df_comp = br_df[br_df['component']==comp]
        last_bug_in_comp = br_df_comp['bug_id'].values[len(br_df_comp.index.values)-1]
        created_on = br_df_comp[br_df_comp['bug_id']==last_bug_in_comp]['created_on'].values[0]
        to_dt = datetime.strptime(created_on, '%Y-%m-%d')
        from_dt = to_dt - dateutil.relativedelta.relativedelta(months=config["fix_period"])

        bug_fixes = {}
        scores = dict((fixer, 0) for fixer in br_df['fixer_names'].values)
        last_fixed_time = ''
        last_fixes = {}
        fixer = ''
        fixed_on = ''
        active_devs = []

        resolved_bugs = []
        list_file = str(last_bug_in_comp) + '_past_bugs.pickle'
        with open('./resources/past_bugs_six_months/' + str(last_bug_in_comp) + '/' + list_file, 'rb') as bugs:
            resolved_bugs = pickle.load(bugs)

        for bug in resolved_bugs:
            history_file = str(bug) + '_history.pickle'
            past_bug_history = {}
            with open('./resources/past_bugs_six_months/history/' + history_file, 'rb') as bugs:
                past_bug_history = pickle.load(bugs)
            bug_history = past_bug_history['bugs'][0]
            for history in bug_history['history']:
                fixed_on = ''
                fixer = ''
                for change in history['changes']:
                    if change['field_name'] == 'resolution' and change['added'] == 'FIXED':
                        first_fixed_on = history['when']
                        first_fixed_on = datetime.strptime(first_fixed_on.value, "%Y%m%dT%H:%M:%S")
                        fixer = history['who'].lower()
                        if first_fixed_on < to_dt and first_fixed_on > from_dt:
                            fixed_on = first_fixed_on
                if last_fixed_time == '' and fixed_on != '':
                    last_fixed_time = fixed_on
                elif last_fixed_time != '' and fixed_on != '':
                    if fixed_on > last_fixed_time:
                        last_fixed_time = fixed_on

                if fixer in bug_fixes:
                    bug_fixes[fixer] = bug_fixes[fixer] + 1
                    if fixed_on != '' and fixed_on > last_fixes[fixer]:
                        last_fixes[fixer] = fixed_on
                else:
                    if fixer != '' and fixed_on != '':
                        bug_fixes[fixer] = 1
                        last_fixes[fixer] = fixed_on
        for fixer_key in bug_fixes.keys():
            if fixer_key in user_dic:
                fixer = user_dic[fixer_key]
                diff = last_fixed_time - last_fixes[fixer_key]
                scores[fixer] = bug_fixes[fixer_key] / math.exp(diff.days/30)
        sorted_devs = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        active_devs = [dev[0] for dev in sorted_devs if dev[1]>0]
        active_dev_set.update(active_devs)

    br_df = br_df[br_df['fixer_names'].isin(active_dev_set)]
    return br_df

user_dic = {}
with open('./trained_models/user_email_name_dic.pickle', 'rb') as f:
    user_dic = pickle.load(f)

config = {}
with open('./resources/config.json') as json_data_file:
    config = json.load(json_data_file)

br_df = pd.read_csv('./resources/eclipse_bugs_data_new.csv')
br_df = br_df.fillna('')

br_df['summary'] = br_df['summary'].str.lower()
br_df['description'] = br_df['description'].str.lower()
br_df['summary'] = br_df['summary'] + ' ' + br_df['description']
br_df['summary'] = br_df['summary'].apply(preprocess_text)
br_df['created_on'] = pd.to_datetime(br_df.created_on)
br_df['fixer'] = br_df['fixer'].str.lower()
br_df['fixer_names'] = br_df['fixer_names'].str.lower()
br_df['class'] = pd.Series(convert_classes(br_df['fixer_names']))

br_df.sort_values('created_on', inplace=True)
br_df['created_on']=br_df['created_on'].dt.strftime('%Y-%m-%d')
br_df.reset_index(inplace=True)
br_df.drop(columns=['index'], inplace=True)

splitted_sets = []
for group, df in br_df.groupby(np.arange(len(br_df)) // 100):
    splitted_sets.append(df)

accuracies = []
precisions = []
recalls = []
top3_accuracies = []
top5_accuracies = []

br_df = splitted_sets[0]
step = 1
for each_set in splitted_sets[1:]:
    br_df_test = each_set
    #br_df = remove_less_active_devs(br_df)

    train_size = len(br_df.index)
    test_size = len(br_df_test.index)

    train_nominal = [br_df.iloc[index][['product', 'component']] for index in range(0, train_size)]
    test_nominal = [br_df_test.iloc[index][['product', 'component']] for index in range(0, test_size)]

    features_train = extract_features(train_nominal, br_df['summary'], True)
    features_test = extract_features(test_nominal, br_df_test['summary'], False)

    classes_train = np.unique(br_df['class'].values)
    classify_liblinear(features_train, br_df['class'].values, True)
    accuracy = classify_liblinear(features_test, br_df_test['class'].values, False)
    top3_accuracy = classify_liblinear_topn(features_test, br_df_test['class'].values, classes_train, False, 3)
    top5_accuracy = classify_liblinear_topn(features_test, br_df_test['class'].values, classes_train, False, 5)

    accuracies.append(accuracy)
    top3_accuracies.append(top3_accuracy)
    top5_accuracies.append(top5_accuracy)    

    br_df = br_df.append(each_set, ignore_index=True, sort=False)
    print(str(step) + 'done')
    step = step+1

print("Top1 accuracy="+str(np.mean(accuracies)))
print("Top3 accuracy="+str(np.mean(top3_accuracies)))
print("Top5 accuracy="+str(np.mean(top5_accuracies)))
