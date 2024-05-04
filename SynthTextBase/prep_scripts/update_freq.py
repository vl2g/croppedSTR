from collections import Counter
import pickle

cnt = 0
filename = '/DATA/ocr_team_2/rajeev/experiments/SynthTextHindi/indic_languages_texts/clean_text_files/mr.txt'
with open(filename, 'r') as f:
    c = Counter()
    for line in f.readlines():
        c += Counter(line.strip())
        cnt += len(line.strip())
        # print c
print(cnt)

for key in c:
    c[key] = float(c[key]) / cnt
    print(key, c[key])

d = dict(c)
# print d
with open("./data/models/char_freq.cp", 'wb') as f:
    pickle.dump(d, f)
