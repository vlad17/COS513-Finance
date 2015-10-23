from gensim.models import Word2Vec, Phrases
from nltk.corpus import brown
import pickle

# set up word2vec
brown_sents = brown.sents()
brown_lower = []
for sentence in brown_sents:
    sentence = [word.lower() for word in sentence]
    brown_lower.append(sentence)

bigram = Phrases(brown_lower)
full_model = Word2Vec(bigram[brown_lower], min_count=1)

with open('models/word2vec', 'wb') as full_model_out:
    pickle.dump(full_model, full_model_out)

with open('models/word2vec_bigram', 'wb') as bigram_out:
    pickle.dump(bigram, bigram_out)