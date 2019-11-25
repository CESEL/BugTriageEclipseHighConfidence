import pandas as pd
import json
from datetime import datetime
import dateutil.relativedelta
from pydriller import RepositoryMining
import math
import operator
from bugzilla import Bugzilla
import collections
import pickle
import re
import numpy as np
import collections

br_df = pd.read_csv('./resources/eclipse_bugs_data_new.csv')
br_df.sort_values('created_on', inplace=True)
br_df.reset_index(inplace=True)
br_df.drop(columns=['index'], inplace=True)
br_df['fixer_names'] = br_df['fixer_names'].str.lower()
fixer_names = np.unique(br_df['fixer_names'].values)

jdt_dict = {}
platform_dict = {}
path = './resources/eclipse_all_bug_comments/'
with open('./resources/jdt_dict.json', 'r') as json_file:
    jdt_dict = json.load(json_file)
with open('./resources/platform_dict.json', 'r') as json_file:
    platform_dict = json.load(json_file)

predictions = []
br_df_len = len(br_df.index)

url = "https://bugs.eclipse.org/bugs/xmlrpc.cgi"
bugzilla = Bugzilla(url)

def predict_by_stacktrace_msr(files, to_dt, id):
    file_found_count = 0
    from_dt = to_dt - dateutil.relativedelta.relativedelta(months=6)
    file_path = ''
    file_repo_path = {}
    scores = dict((fixer, 0) for fixer in fixer_names)
    with open(path+id+'/file_package_repo_path.json', 'r') as json_file:
        file_repo_path = json.load(json_file)
    for file in files:
        if file in file_repo_path:
            repo_path = file_repo_path[file][0]
            file_path = file_repo_path[file][1]
            high_level_packages = re.sub(repo_path, '', file_path).split('/')
            high_level_packages = [package for package in high_level_packages if package != '']
            if len(high_level_packages) > 0:
                file_found_count = file_found_count + 1
                high_level_package = repo_path+'/'+high_level_packages[0]+'/'
                last_commit_date = ''
                commiters = {}
                last_commits = {}
                for commit in RepositoryMining(path_to_repo=repo_path, filepath=high_level_package, since=from_dt, to=to_dt, only_no_merge=True, reversed_order=True).traverse_commits():
                    if last_commit_date == '':
                        last_commit_date = commit.committer_date
                    commiter_name = commit.author.name.lower()
                    if re.search('.*bug.*', commit.msg.lower()):
                        if commiter_name in fixer_names :
                            if commiter_name in commiters:
                                commiters[commiter_name] = commiters[commiter_name] + 1
                            else:
                                commiters[commiter_name] = 1
                                last_commits[commiter_name] = commit.committer_date
            for dev in fixer_names:
                if dev in commiters:
                    diff = last_commit_date - last_commits[dev]
                    scores[dev] = scores[dev] + commiters[dev] / math.exp(diff.days/30)
    if file_found_count != 0:
        scores = {k: v / file_found_count for k, v in scores.items()}
    return scores

def collect_msr_scores():
    scores_list = []
    for index in range(0, br_df_len):
        id = str(br_df.iloc[index]['bug_id'])
        product = br_df.iloc[index]['product']
        component = br_df.iloc[index]['component']
        created_on = br_df.iloc[index]['created_on']
        to_dt = datetime.strptime(created_on, '%Y-%m-%d')
        files = []

        # check whether stack trace exists
        file = ''
        with open(path + id + '/' + id + '_traces.json', 'r') as trace:
            trace_json = json.load(trace)
        key_size = len(trace_json.keys())
        for key in range(0, key_size):
            frame = trace_json[str(key)][0]
            if frame.startswith('org.eclipse'):
                file = re.findall('\(.+\)', frame)[0]
                file = re.sub('\(|:.*', '', file)
                file = id + '-' + file
                files.append(file)

        if len(files) > 0:
            if len(files) > 6:
                files = files[0:6]
            files = np.unique(files)
            stacktrace_score = predict_by_stacktrace_msr(files, to_dt, id)
            scores_list.append(stacktrace_score)
        else:
            scores = dict((fixer, 0) for fixer in fixer_names)
            repo = ''
            if product == 'JDT':
                if component in jdt_dict:
                    repo = jdt_dict[component]
            else:
                if component in platform_dict:
                    repo = platform_dict[component]
            if repo != '':
                from_dt = to_dt - dateutil.relativedelta.relativedelta(months=6)
                commits = {}
                last_commits = {}
                last_commit_date = ''
                for commit in RepositoryMining(path_to_repo=repo, since=from_dt, to=to_dt, only_no_merge=True, reversed_order=True).traverse_commits():
                    if last_commit_date == '':
                        last_commit_date = commit.committer_date
                    author = commit.author.name.lower()
                    if author in fixer_names:
                        if re.search('.*bug.*', commit.msg):
                            if author in commits:
                                commits[author] = commits[author] + 1
                            else:
                                commits[author] = 1
                                last_commits[author] = commit.committer_date

                for dev in fixer_names:
                    if dev in commits:
                        diff = last_commit_date - last_commits[dev]
                        scores[dev] = commits[dev]/math.exp(diff.days/30)
            scores_list.append(scores)
        print(id+ ' Done')
    return scores_list

scores_list = collect_msr_scores()
scores_df = pd.DataFrame(scores_list)
print(len(scores_df.index))
print(len(scores_df.columns))
with open('./trained_models/score_dataframes/msr_score_df_six_months.pickle', 'wb') as f:
    pickle.dump(scores_df, f)

scores_df = pd.DataFrame()
with open('./trained_models/score_dataframes/msr_score_df_six_months.pickle', 'rb') as f:
    scores_df = pickle.load(f)
print(len(scores_df.columns))
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
        if sorted_devs[0][1] > 0:
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
        else:
            accuracy_top1.append(0)
            accuracy_top3.append(0)
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
