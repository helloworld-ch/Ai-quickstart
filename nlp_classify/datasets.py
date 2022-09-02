import os
import re
from torch.utils.data import DataLoader,Dataset
import torch
from lib import ws,train_batch_size,max_len

# 分词
def tokenlize(content):
    content = re.sub("<.*?>"," ",content)
    # filters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?',
    #            '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    #
    # content = re.sub('|'.join(filters), ' ', content)

    content = re.sub('[0-9!"#$%&()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+'," ",content)
    token = [i.strip().lower() for i in content.split()]
    return token

class ImdbDataset(Dataset):
    def __init__(self,train = True):
        self.train_path = r"E:\AI\data\aclImdb_v1\aclImdb\train"
        self.test_path = r"E:\AI\data\aclImdb_v1\aclImdb\test"
        data_path = self.train_path if train else self.test_path

        # 将所有文件放入列表
        temp_data_path = [os.path.join(data_path,"pos"),os.path.join(data_path,"neg")]
        self.total_file_path = []
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path,i) for i in file_name_list]
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, index):
        file_path = self.total_file_path[index]
        # 定义标签
        label_str = file_path.split("\\")[-2]
        label = 0 if label_str == "neg" else 1 # 负面的为0，正面为1
        content = tokenlize(open(file_path,encoding='utf-8').read())
        return label,content

    def __len__(self):
        return len(self.total_file_path)
        pass

def get_data_loader(train = True,batch_size = train_batch_size):
    datasets = ImdbDataset(train)
    dataloader = DataLoader(datasets,shuffle=True,batch_size=batch_size,collate_fn=collate_fn)
    return dataloader

# 自定义的一个内容与标签绑定
def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    labels = torch.tensor(batch[0], dtype=torch.int32)
    texts = batch[1]
    del batch
    texts = [ws.transform(text,max_len = max_len)for text in texts]
    texts = torch.LongTensor(texts)
    labels = torch.LongTensor(labels.numpy())
    return texts, labels

if __name__ == '__main__':
    for step,(x,y) in enumerate(get_data_loader()):
        print(x)
        print()
        print(y)
        break
