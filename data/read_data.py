import json

data = json.load(open('data.json', 'r'))
for each in data:
    if each is None:
        continue
    print(each['url'])
    print(each['question'])
    print(each['ans'])
    print()
