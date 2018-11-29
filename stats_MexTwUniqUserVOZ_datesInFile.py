import glob
import gzip
import json
import os
from collections import defaultdict

dates = defaultdict(list)

for fn in glob.glob('stats/AggMexTwUniqUserVOZ/*.json.gz'):
    print(fn)
    with gzip.open(fn) as fin:
        data = json.load(fin)
        for d in data.keys():
            dates[d].append(os.path.basename(fn).replace('.json.gz', ''))

with open('stats/AggMexTwUniqUserVOZ/dates_in_file.json', 'w') as fw:
    json.dump(dates, fw)
