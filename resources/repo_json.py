import json

def populate_jdt_dict():
    jdt_dic = {}
    jdt_dic['Core'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.jdt.core'
    jdt_dic['Debug'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.jdt.debug'
    jdt_dic['UI'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.jdt.ui'

    with open('./resources/jdt_dict.json', 'w') as json_file:
        json.dump(jdt_dic, json_file)

def populate_platform_dict():
    platform_dict = {}
    platform_dict['Compare'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.team'
    platform_dict['Debug'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.debug'
    platform_dict['Resources'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.resources'
    platform_dict['Releng'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.releng'
    platform_dict['Runtime'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.runtime'
    platform_dict['SWT'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.swt'
    platform_dict['Team'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.team'
    platform_dict['Text'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.text'
    platform_dict['UI'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.ui'
    platform_dict['User Assistance'] = '/home/aindrila/Documents/Projects/eclipse/eclipse.platform.ua'

    with open('./resources/platform_dict.json', 'w') as json_file:
        json.dump(platform_dict, json_file)

populate_platform_dict()
populate_jdt_dict()