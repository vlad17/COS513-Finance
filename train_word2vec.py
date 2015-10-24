from gensim.models import Word2Vec, Phrases
from nltk.corpus import brown
from nltk.corpus import stopwords
import nltk
import pickle

nltk.data.path.append("/n/fs/gcf")
stops = stopwords.words("english")

# set up word2vec
brown_sents = brown.sents()
brown_lower = []
for sentence in brown_sents:
    sentence = [word.lower() for word in sentence]
    sentence = [word for word in sentence if word not in stops]
    brown_lower.append(sentence)

bigram = Phrases(brown_lower)
b = bigram[brown_lower]
trigram = Phrases(b)
t = trigram[b]
quadgram = Phrases(t)
q = quadgram[t]
full_model = Word2Vec(q, min_count=1)
full_model.init_sims(replace=True)

with open('models/word2vec', 'wb') as full_model_out:
    pickle.dump(full_model, full_model_out)

with open('models/word2vec_bigram', 'wb') as bigram_out:
    pickle.dump(bigram, bigram_out)

with open('models/word2vec_trigram', 'wb') as trigram_out:
    pickle.dump(trigram, trigram_out)

with open('models/word2vec_quadgram', 'wb') as quadgram_out:
    pickle.dump(quadgram, quadgram_out)
