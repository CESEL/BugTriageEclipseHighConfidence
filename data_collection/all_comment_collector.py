from bugzilla import Bugzilla
import pandas as pd
import datetime
import os
import shutil
import numpy as np
import shutil

url = "https://bugs.eclipse.org/bugs/xmlrpc.cgi"
location = './eclipse_all_bug_comments/'
os.mkdir(location)

all_bugs = pd.read_csv('./resources/Eclipse_Bugs.csv')['Bug ID'].values
bugzilla = Bugzilla(url)
bugs = bugzilla.getbugs(all_bugs.tolist(), include_fields=['creation_time'])

for bug in bugs:
    all_comments = ''
    id = str(bug.id)
    path = location+id
    os.mkdir(path)

    try:
        for comment in bug.getcomments():
            all_comments = all_comments + comment['text'] + '\n'
    except Exception:
        print(id+' error')
    print(id+' Done')

    with open(path+'/' + id + '_comments.txt', 'w', encoding='utf-8') as log:
        log.write(all_comments)


