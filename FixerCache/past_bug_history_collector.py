import pandas as pd
import json
from datetime import datetime
import dateutil.relativedelta
from pydriller import RepositoryMining
import math
import operator
from bugzilla import Bugzilla
import collections
import os
import pickle
import shutil
import logging

def collect_past_bugs():
    url = "https://bugs.eclipse.org/bugs/xmlrpc.cgi"
    bugzilla = Bugzilla(url)
    bug_ids = pd.read_csv('./resources/Eclipse_Bugs.csv')['Bug ID'].values
    all_bugs = bugzilla.getbugs(bug_ids.tolist(), include_fields=['creation_time', 'product', 'component'])

    for bug in all_bugs:
        id = str(bug.id)
        product = bug.product
        component = bug.component
        created_on = bug.creation_time
        to_dt = datetime.strptime(created_on.value, "%Y%m%dT%H:%M:%S")
        # months indicate the caching period
        from_dt = to_dt - dateutil.relativedelta.relativedelta(months=6)

        start = from_dt.strftime("%Y-%m-%d")
        end = to_dt.strftime("%Y-%m-%d")
        query_url = 'https://bugs.eclipse.org/bugs/buglist.cgi?' \
                    'bug_status=RESOLVED&bug_status=VERIFIED&bug_status=CLOSED' \
                    '&chfield=resolution&chfieldfrom=' + start + '&chfieldto=' + end + \
                    '&chfieldvalue=FIXED&classification=Eclipse&component=' + component + '&product=' + product + \
                    '&query_format=advanced&resolution=FIXED'

        url = "https://bugs.eclipse.org/bugs/xmlrpc.cgi"
        query = bugzilla.url_to_query(query_url)
        query["include_fields"] = ["id"]
        resolved_bugs = bugzilla.query(query)
        resolved_bugs = [bug.id for bug in resolved_bugs]

        os.mkdir('/home/aindrila/Documents/Projects/past_bugs_six_months/' + id)
        file = id + '_past_bugs.pickle'
        with open('/home/aindrila/Documents/Projects/past_bugs_six_months/' + id + '/' + file, 'wb') as out_file:
            pickle.dump(resolved_bugs, out_file)
        print(id + ' Done')

def collect_past_bugs_history():
    url = "https://bugs.eclipse.org/bugs/xmlrpc.cgi"
    bugzilla = Bugzilla(url)
    bug_ids = pd.read_csv('./resources/Eclipse_Bugs.csv')['Bug ID'].values
    past_bug_set = set()
    for id in bug_ids:
        id = str(id)
        file = id + '_past_bugs.pickle'
        past_bugs = []
        with open('/home/aindrila/Documents/Projects/past_bugs_six_months/' + id + '/' + file, 'rb') as bugs:
            past_bugs = pickle.load(bugs)
        past_bug_set.update(past_bugs)

    past_bug_list = list(past_bug_set)
    resolved_bugs = bugzilla.getbugs(past_bug_list, include_fields=['id'])

    for resolved_bug in resolved_bugs:
        bug_history = resolved_bug.get_history_raw()
        file = str(resolved_bug.id)+'_history.pickle'
        with open('/home/aindrila/Documents/Projects/past_bugs_six_months/history/'+file, 'wb') as out_file:
            pickle.dump(bug_history, out_file)
        print(str(resolved_bug.id) + ' Done')

collect_past_bugs()
collect_past_bugs_history()


