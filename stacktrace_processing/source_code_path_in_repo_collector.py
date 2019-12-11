import pandas as pd
import re
from nltk.corpus import wordnet
import json
import glob
from datetime import datetime
import dateutil.relativedelta
from pydriller import RepositoryMining
import collections
import numpy as np
import math
import operator

# the csv contains the bug dataset
bug_reports = pd.read_csv('./resources/eclipse_bugs_data.csv')

# This path contains the processed stack traces
path = './resources/eclipse_all_bug_comments/'
df_len = len(bug_reports.index)
developers = []
comps = []
count = 0

jdt_dict = {}
platform_dict = {}
with open('./resources/jdt_dict.json', 'r') as json_file:
    jdt_dict = json.load(json_file)
with open('./resources/platform_dict.json', 'r') as json_file:
    platform_dict = json.load(json_file)

for index in range(0,df_len):
    id = str(bug_reports.iloc[index]['bug_id'])
    product = bug_reports.iloc[index]['product']
    component = bug_reports.iloc[index]['component']
    trace_json = {}
    files = []
    packages = []
    package = ''
    file_path = ''
    package_path = ''
    authors = []
    file_repo_path_json = {}
    with open(path+id+'/'+id+'_traces.json', 'r') as trace:
        trace_json = json.load(trace)
    key_size = len(trace_json.keys())

    # Get the source code file names
    for key in range(0,key_size):
        frame = trace_json[str(key)][0]
        if frame.startswith('org.eclipse'):
            file = re.findall('\(.+\)', frame)[0]
            file = re.sub('\(|:.*', '', file)
            files.append(file)

    if files != []:
        repo = ''
        repo_path = ''
        if product == 'JDT':
            if component in jdt_dict:
            	repo = jdt_dict[component]
        else:
            if component in platform_dict:
            	repo = platform_dict[component]        

        # Get the location of the source code file in the repository
        
        for file in files:
            for code_file in glob.iglob(repo + '/**/' + file, recursive=True):
        	file_path = code_file
        	repo_path = repo
        	bug_file = id + '-' + file
        	file_repo_path_json[bug_file] = [repo_path, file_path]                            

    with open(path+id+'/file_package_repo_path.json', 'w') as json_file:
        json.dump(file_repo_path_json, json_file)

    print(id+' Done')
print(count)
