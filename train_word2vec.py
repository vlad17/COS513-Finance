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

full_model_out = open('models/word2vec', 'w')
bigram_out = open('models/word2vec_bigram', 'w')

pickle.dump(full_model, full_model_out)
pickle.dump(bigram, bigram_out)