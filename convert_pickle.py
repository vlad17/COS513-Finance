import sys
import pickle

with open(sys.argv[1], 'rb') as model_file:
    model = pickle.load(model_file)

with open('./1000_2.model', 'wb') as model_out:
    pickle.dump(model, model_out, protocol=2)