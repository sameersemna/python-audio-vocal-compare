#!/usr/bin/python

import sys
from csv import reader
import json
import urllib.parse
import pandas as pd
import traceback
from youtubesearchpython import ChannelSearch,ResultMode

# print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

searchText = sys.argv[1]
folderRoot = sys.argv[2]
csvChannels = folderRoot + '/youtubeChannels.csv'
# csvjson -k name -d ',' ~/Private/Work/Projects/Websites/Islam/shamela.org/src/assets/csv/youtubeChannels.csv > ~/Private/Work/Projects/Websites/Islam/shamela.org/src/api/youtubeChannels.json
results = []

def flattenjson(b, delim):
    val = {}
    for i in b.keys():
        if isinstance(b[i], dict):
            get = flattenjson(b[i], delim)
            for j in get.keys():
                val[i + delim + j] = get[j]
        else:
            val[i] = b[i]

    return val

def flatten_json_org(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            if isinstance(x, str) and not x.isnumeric():
                out[name[:-1]] = x

    flatten(y)
    return out


with open(csvChannels, 'r', encoding='utf-8') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    header = next(csv_reader)

    if header != None:
        for row in csv_reader:
            # print(row)
            channelName, channelId, channelUrl = row[0], row[1], row[2]
            channelName = u' '.join((channelName)).encode('ascii', 'ignore').decode('ascii')
            if channelName[0:2] == '//':
                print('Ignored: ' + channelName + ' | ' + channelId)
            else:
                print('Processing:' + channelName + ' | ' + channelId)
                try:
                    print('Processing 1:' + channelName + ' | ' + searchText)
                    search = ChannelSearch(searchText, channelId)
                    print('Processing 1.1:' + channelName + ' | ' + channelId)
                    print(search.result(mode = ResultMode.json))
                    resultJson = search.result(mode = ResultMode.json)
                    resultJson = json.loads(resultJson)
                    resultJson = resultJson['result']
                    print('Processing 2:' + channelName + ' | ' + channelId)

                    # resultJson = flattenjson(resultJson, "__")
                    resultJson = flatten_json(resultJson)
                    resultJson = list(map(flatten_json, resultJson))
                    print(resultJson)

                    results = results + resultJson
                    # sys.exit()
                except Exception as e:
                    print('An exception occurred:' + channelName)
                    print(e)
                    traceback.print_exc()
                    sys.exit()

        # print('Header was:')
        # print(header)

# search = ChannelSearch('Watermelon Sugar', "UCZFWPqqPkFlNwIxcpsLOwew")
# print(search.result(mode = ResultMode.json))

# df = pd.DataFrame.from_dict(resultJson, orient="index")
df = pd.DataFrame(results)
# df.to_csv('csvfile.csv', encoding='utf-8', index=False)
# df.to_json('csvfile.json', orient="split", index=False)
df.to_json(folderRoot + '/cache/salafisearch_' + urllib.parse.quote(searchText) + '.json', orient="records")

print('Search completed for:' + searchText)

# ['/home/ubuntu/Projects/Shamela/www/scripts', '/usr/lib/python36.zip', '/usr/lib/python3.6', '/usr/lib/python3.6/lib-dynload',
#     '/home/ubuntu/.local/lib/python3.6/site-packages', '/usr/local/lib/python3.6/dist-packages', '/usr/lib/python3/dist-packages']

# ['/home/ubuntu/Projects/Shamela/www/scripts', '/usr/lib/python36.zip', '/usr/lib/python3.6',
#     '/usr/lib/python3.6/lib-dynload', '/usr/local/lib/python3.6/dist-packages', '/usr/lib/python3/dist-packages']

# import sys
# sys.path.append('/home/ubuntu/.local/lib/python3.6/site-packages')
# print(sys.path)
