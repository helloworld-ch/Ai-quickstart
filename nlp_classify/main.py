import pickle
import os
from datasets import tokenlize
from word_Sequence import Word2Sequence
from tqdm import tqdm

if __name__ == '__main__':
    path = r"E:\AI\data\aclImdb_v1\aclImdb\train"
    ws = Word2Sequence()
    temp_data_path = [os.path.join(path, "pos"), os.path.join(path, "neg")]
    for data_path in temp_data_path:
        file_paths = [os.path.join(data_path,file_name) for file_name in os.listdir(data_path)]
        for file_path in tqdm(file_paths):
            sentence = tokenlize(open(file_path,encoding='utf-8').read())
            ws.fit(sentence)
    ws.build_vocab(min = 10,max_feature=10000)
    pickle.dump(ws,open("./models/ws.pkl",'wb'))
    print(len(ws))
    print(ws.to_word(20))
