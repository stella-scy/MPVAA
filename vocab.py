import sys
import pickle 

fp = open('total_new_path.txt')
data = fp.read()
#print('data', data)
words = data.split()
fp.close()

unwanted_chars = ".,-_ (and so on)"
wordfreq = {}
for raw_word in words:
    word = raw_word.strip(unwanted_chars)
    if word not in wordfreq:
        wordfreq[word] = 0 
    wordfreq[word] += 1


print('len(wordfreq.keys())', len(wordfreq.keys()))
with open('my.vocab', 'w') as f:
    for key, value in wordfreq.items():
        f.write('%s %s\n' % (key, value))
