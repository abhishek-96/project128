import pickle
import json
from pprint import pprint
headers={}
desc={}
i=0
with open('sample-1M.jsonl', 'rb') as f:
    # result = json.load(f)
    for line in f:
        data = json.loads(line)
        print i
        desc.update({i:data['content'].encode('utf-8')})
        headers.update({i:data['title'].encode('utf-8')})
        i = i + 1
        if i == 100000:
            break
        
print len(headers)
print len(desc)
file=open('tokenstest2000.pkl','wb')
pickle.dump((headers,desc),file)
print 'Done Pickling'
file.close()
