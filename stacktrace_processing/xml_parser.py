from xml.etree import ElementTree as et
import pandas as pd
import glob
import json
import os

all_bugs = pd.read_csv('.././resources/Eclipse_Bugs.csv')['Bug ID'].values
path = '.././resources/eclipse_all_bug_comments/'

def parse_stack_traces():
    for bug in all_bugs:
        bug = str(bug)
        folder = path+bug+'/'
        xml_output = glob.glob(folder+'*.xml')
        traces = {}
        if len(xml_output) > 0:
            tree = et.parse(xml_output[0])
            root = tree.getroot()
            for trace in root.iter('Frame'):
                depth = trace.attrib['depth']
                if depth in traces:
                    traces[depth].append(trace.text)
                else:
                    file = []
                    file.append(trace.text)
                    traces[depth] = file

        with open(path+bug+'/'+bug+'_traces.json', 'w') as fp:
            json.dump(traces, fp)
        print(bug+' Done')

parse_stack_traces()



