import sys
import pickle

with open(sys.argv[1], 'rb') as model_file:
    model = pickle.load(model_file)

pickle.dump(model, './1000_2.model', protocol=2)