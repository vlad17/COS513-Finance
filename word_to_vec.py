from gensim.models import word2vec, Phrases
import nltk
from nltk.corpus import brown
import gensim



brown_sents = brown.sents()
brown_lower = []
for sentence in brown_sents:
    sentence = [word.lower() for word in sentence]
    brown_lower.append(sentence)
    
bigram = Phrases(brown_lower)
full_model = gensim.models.Word2Vec(bigram[brown_lower], min_count=1)

