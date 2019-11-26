from bugzilla import Bugzilla
import collections
import json
from _datetime import datetime
import dateutil.relativedelta
import pandas as pd
import math
import operator
import glob
import pickle
import logging
import numpy as np

br_df = pd.read_csv('.././resources/eclipse_bugs_data_new.csv')
br_df['created_on'] = pd.to_datetime(br_df.created_on)
br_df.sort_values('created_on', inplace=True)
br_df['created_on']=br_df['created_on'].dt.strftime('%Y-%m-%d')
br_df.reset_index(inplace=True)
br_df.drop(columns=['index'], inplace=True)
br_df['fixer'] = br_df['fixer'].str.lower()
br_df['fixer_names'] = br_df['fixer_names'].str.lower()
fixers_names = br_df['fixer_names'].values
br_df_len = len(br_df.index)

user_dic = {}
with open('.././trained_models/user_email_name_dic.pickle', 'rb') as f:
    user_dic = pickle.load(f)

def collect_component_cache_scores():
    scores_list = []
    for index in range(0, br_df_len):
        id = str(br_df.iloc[index]['bug_id'])
        product = br_df.iloc[index]['product']
        component = br_df.iloc[index]['component']
        created_on = br_df.iloc[index]['created_on']
        to_dt = datetime.strptime(created_on, '%Y-%m-%d')
        from_dt = to_dt - dateutil.relativedelta.relativedelta(months=6)

        bug_fixes = {}
        last_fixed_time = None
        last_fixes = {}
        fixer = None
        fixed_on = None
        scores = dict((fixer, 0) for fixer in fixers_names)

        resolved_bugs = []
        list_file = id + '_past_bugs.pickle'
        with open('.././resources/past_bugs_six_months/' + id + '/' + list_file, 'rb') as bugs:
            resolved_bugs = pickle.load(bugs)

        # calculation of the scores of the developers in the cache
        for bug in resolved_bugs:
            history_file = str(bug) + '_history.pickle'
            past_bug_history = {}
            with open('.././resources/past_bugs_six_months/history/' + history_file, 'rb') as bugs:
                past_bug_history = pickle.load(bugs)
            bug_history = past_bug_history['bugs'][0]
            for history in bug_history['history']:
                fixed_on = None
                fixer = None
                for change in history['changes']:
                    if change['field_name'] == 'resolution' and change['added'] == 'FIXED':
                        first_fixed_on = history['when']
                        first_fixed_on = datetime.strptime(first_fixed_on.value, "%Y%m%dT%H:%M:%S")
                        fixer = history['who'].lower()
                        if first_fixed_on < to_dt and first_fixed_on > from_dt:
                            fixed_on = first_fixed_on
                if last_fixed_time is None and fixed_on is not None:
                    last_fixed_time = fixed_on
                elif last_fixed_time is not None and fixed_on is not None:
                    if fixed_on > last_fixed_time:
                        last_fixed_time = fixed_on

                if fixer in bug_fixes:
                    bug_fixes[fixer] = bug_fixes[fixer] + 1
                    if fixed_on is not None and fixed_on > last_fixes[fixer]:
                        last_fixes[fixer] = fixed_on
                elif fixer:
                    if fixed_on is not None:
                        bug_fixes[fixer] = 1
                        last_fixes[fixer] = fixed_on
        for fixer_key in bug_fixes.keys():
            if fixer_key in user_dic:
                fixer = user_dic[fixer_key]
                diff = last_fixed_time - last_fixes[fixer_key]
                scores[fixer] = bug_fixes[fixer_key] / math.exp(diff.days/30)
        scores_list.append(scores)
        print(str(id)+ ' Done')
    return scores_list

scores_list = collect_component_cache_scores()
scores_df = pd.DataFrame(scores_list)
print(len(user_dic.keys()))
print(len(scores_df.index))
print(len(scores_df.columns))
with open('.././trained_models/score_dataframes/component_score_df_six_months.pickle', 'wb') as f:
    pickle.dump(scores_df, f)

with open('.././trained_models/score_dataframes/component_score_df_six_months.pickle', 'rb') as f:
    scores_df = pickle.load(f)
splitted_sets = []
for group, df in br_df.groupby(np.arange(len(br_df)) // 100):
    splitted_sets.append(df)

br_df = splitted_sets[0]
accuracy_top1 = []
accuracy_top3 = []
accuracy_top5 = []
for each_set in splitted_sets[1:]:
    br_df_test = each_set
    fixers = np.unique(br_df['fixer_names'].values)
    for index in each_set.index.to_list():
        scores = scores_df.iloc[index][fixers].to_dict()
        sorted_devs = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        top3 = 3
        top5 = 5
        if len(sorted_devs) < 3:
            top3 = len(sorted_devs)
        if len(sorted_devs) < 5:
            top5 = len(sorted_devs)
        else:
            top5 = len(sorted_devs)
        top3_devs = [dev[0] for dev in sorted_devs[0:top3]]
        top5_devs = [dev[0] for dev in sorted_devs[0:top5]]
        if sorted_devs[0][0] == each_set.loc[index]['fixer_names']:
            accuracy_top1.append(1)
        else:
            accuracy_top1.append(0)
        if each_set.loc[index]['fixer_names'] in top3_devs:
            accuracy_top3.append(1)
        else:
            accuracy_top3.append(0)
        if each_set.loc[index]['fixer_names'] in top5_devs:
            accuracy_top5.append(1)
        else:
            accuracy_top5.append(0)

    br_df = br_df.append(each_set, ignore_index=True, sort=False)

counter = collections.Counter(accuracy_top1)
top1_accuracy = counter[1]/len(accuracy_top1)

counter = collections.Counter(accuracy_top3)
top3_accuracy = counter[1]/len(accuracy_top3)

counter = collections.Counter(accuracy_top5)
top5_accuracy = counter[1]/len(accuracy_top5)

print(top1_accuracy)
print(top3_accuracy)
print(top5_accuracy)
