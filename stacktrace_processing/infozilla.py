import numpy as np
import os
import subprocess
import pandas as pd
import glob

# Replace the path with the actual location
path = '/home/aindrila/Documents/Projects/bugtriagingoss/resources/eclipse_all_bug_comments/'
# The csv contains all the bug ids
all_bugs = pd.read_csv('./resources/Eclipse_Bugs.csv')['Bug ID'].values
# Put the path where infozilla tool is running
os.chdir('/home/aindrila/Documents/Projects/infozilla_tool/infozilla')

# The comments of the bug reports should be downloaded
for bug in all_bugs:
    bug = str(bug)
    comment_file = ''
    files = glob.glob(path+bug+'/'+bug+'_comments.txt')
    if len(files) > 0:
        comment_file = files[0]
        command = 'gradle run --args="'+comment_file+'"'
        print(command)
        try:
            subprocess.call(command, shell=True)
        except Exception:
            print(bug+' error')
        print(bug+' success')


