from bugzilla import Bugzilla
import pandas as pd
import datetime
import os
import shutil
import numpy as np
import logging

def collect_bug_reports():
    logging.basicConfig(filename='bugtriaging.log', level=logging.DEBUG)
    url = "https://bugs.eclipse.org/bugs/xmlrpc.cgi"
    bugzilla = Bugzilla(url)
    bug_ids = pd.read_csv('./resources/Eclipse_Bugs.csv')['Bug ID'].values
    all_bugs = bugzilla.getbugs(bug_ids.tolist(), include_fields=['summary', 'reporter', 'assigned_to', 'creation_time', 'product', 'component'])
    bug_reports = []

    for bug in all_bugs:
        id = str(bug.id)
        bug_data = {}
        bug_data['product'] = bug.product
        bug_data['component'] = bug.component
        bug_data['assignee'] = bug.assigned_to
        created_on = bug.creation_time
        created_on = datetime.datetime.strptime(created_on.value, "%Y%m%dT%H:%M:%S")
        bug_data['created_on'] = created_on.strftime('%Y-%m-%d')
        bug_data['summary'] = bug.summary
        bug_data['reporter'] = bug.reporter

        bug_history = bug.get_history_raw()['bugs'][0]
        fixer = ''
        for history in bug_history['history']:
            for change in history['changes']:
                if change['field_name'] == 'resolution' and change['added'] == 'FIXED':
                    fixed_on = history['when']
                    fixer = history['who']
        if fixer != '' and bug_data['assignee'] != fixer:
            bug_data['assignee'] = fixer
        fixed_on = datetime.datetime.strptime(fixed_on.value, "%Y%m%dT%H:%M:%S")
        bug_data['fixed_on'] = fixed_on.strftime('%Y-%m-%d')

        bug_data['description'] = ''
        try:
            all_comments = bug.getcomments()
            if len(all_comments) > 0:
                bug_data['description'] = all_comments[0]['text']

        except Exception:
            print('Error '+ id)

        bug_reports.append((id, bug_data['created_on'], bug_data['summary'], bug_data['description'], bug_data['product'], bug_data['component'], bug_data['reporter'], bug_data['assignee']))
        logging.info('Done '+id)

    df = pd.DataFrame(bug_reports, columns=('bug_id','created_on', 'summary', 'description', 'product', 'component', 'reporter', 'fixer'))
    df.to_csv('./resources/eclipse_bugs_data_new.csv')

def collect_fixer_names():
    user_dic = {}
    bug_data = pd.read_csv('./resources/eclipse_bugs_data_new.csv')
    fixers = bug_data['fixer'].values.tolist()
    url = "https://bugs.eclipse.org/bugs/xmlrpc.cgi"
    bugzilla = Bugzilla(url)
    users = bugzilla.getusers(fixers)
    for user in users:
        user_dic[user.email] = user.real_name.lower()
    fixer_names = [user_dic[fixer] for fixer in fixers]
    bug_data['fixer_names'] = fixer_names
    bug_data.to_csv('./resources/eclipse_bugs_data_new.csv')

collect_bug_reports()
collect_fixer_names()






