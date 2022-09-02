import pickle

ws = pickle.load(open("./models/ws.pkl","rb"))
train_batch_size = 128
test_batch_size = 1000
max_len = 20