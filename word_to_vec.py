from gensim.models import word2vec, Phrases
import nltk
from nltk.corpus import brown
import gensim



sentences = brown.sents()
bigram = Phrases(sentences)
full_model = gensim.models.Word2Vec(bigram[sentences], min_count=1)

