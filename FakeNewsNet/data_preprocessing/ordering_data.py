import json
import pandas as pd
from datetime import datetime

with open('../data/real_propagation_paths.jsonl') as f:
    data = [json.loads(line) for line in f]

for sublist in data:
    sublist.sort(key=lambda x: datetime.strptime(x['created_at'], "%a %b %d %H:%M:%S %z %Y"))

with open('../data/ordered_real_propagation_paths.jsonl', 'w') as f:
    for sublist in data:
        f.write(json.dumps(sublist) + '\n')

print('Finished processing real propagation paths. Now processing fake propagation paths.')

with open('../data/fake_propagation_paths.jsonl') as f:
    data = [json.loads(line) for line in f]

for sublist in data:
    sublist.sort(key=lambda x: datetime.strptime(x['created_at'], "%a %b %d %H:%M:%S %z %Y"))

with open('../data/ordered_fake_propagation_paths.jsonl', 'w') as f:
    for sublist in data:
        f.write(json.dumps(sublist) + '\n')