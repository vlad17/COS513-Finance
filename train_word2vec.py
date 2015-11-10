from gensim.models import Word2Vec, Phrases
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import stopwords
import nltk
import pickle
import fileinput

nltk.data.path.append("/n/fs/gcf")
stops = stopwords.words("english")
'''
# set up word2vec brown
brown_sents = brown.sents()
brown_lower = []
for sentence in brown_sents:
    sentence = [word.lower() for word in sentence]
    sentence = [word for word in sentence if word not in stops]
    brown_lower.append(sentence)

# set up word2vec reuters
reuters_sents = brown.sents()
reuters_lower = []
for sentence in reuters_sents:
    sentence = [word.lower() for word in sentence]
    sentence = [word for word in sentence if word not in stops]
    print(*sentence, sep="")    
    reuters_lower.append(sentence)    
'''
# set up europarl english    
europarl_lower = []
with open('/n/fs/gcf/europarl_en.txt','r') as f:
	for line in f:
		line = line.strip()
		sentence = line.split()
		sentence = [p.lower() for p in sentence]
		sentence = [p for p in sentence if p not in stops]
		#print(sentence[1])
		europarl_lower.append(sentence)
w2v = europarl_lower #try using reuters corpus



bigram = Phrases(w2v)
b = bigram[w2v]
trigram = Phrases(b)
t = trigram[b]
quadgram = Phrases(t)
q = quadgram[t]
full_model = Word2Vec(q, min_count=1, window=5, workers=2)
full_model.init_sims(replace=True)

with open('/n/fs/gcf/COS513-Finance/models/word2vec', 'wb') as full_model_out:
    pickle.dump(full_model, full_model_out)

with open('/n/fs/gcf/COS513-Finance/models/word2vec_bigram', 'wb') as bigram_out:
    pickle.dump(bigram, bigram_out)

with open('/n/fs/gcf/COS513-Finance/models/word2vec_trigram', 'wb') as trigram_out:
    pickle.dump(trigram, trigram_out)

with open('/n/fs/gcf/COS513-Finance/models/word2vec_quadgram', 'wb') as quadgram_out:
    pickle.dump(quadgram, quadgram_out)
