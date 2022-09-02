class Word2Sequence():
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"

    UNK = 0
    PAD = 1
    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {} # 统计词频
        self.fited = False

    def to_index(self, word):
        """word -> index"""
        assert self.fited == True, "必须先进行fit操作"
        return self.dict.get(word, self.UNK)

    def to_word(self, index):
        """index -> word"""
        assert self.fited, "必须先进行fit操作"
        if index in self.inverse_dict:
            return self.inverse_dict[index]
        return self.UNK_TAG

    def fit(self,sentence):
        """
        把词传入词典
        :param sentence: 【word1,word2...】
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word,0)+1
        self.fited = True

            # if word not in self.count:
            #     self.count[word] = 0
            # self.count+=1

    def build_vocab(self,min = None,max = None,max_feature = None):
        """
        生成词典
        :param min:
        :param max:
        :param max_feature:
        :return:
        """
        # 过滤词频低的词
        if min is not None and isinstance(min,int):
            self.count = {word:self.count.get(word) for word in self.count if self.count.get(word)>min}

        # 过滤高词频的词
        if max is not None and isinstance(max,int):
            self.count = {word:self.count.get(word) for word in self.count if self.count.get(word)<max}

        # 保留一定词
        if max_feature is not None and isinstance(max_feature,int):
            self.count = dict(sorted(self.count.items(),key=lambda x:x[-1],reverse=True)[:max_feature])

        # 放入词典
        for word in self.count:
            self.dict[word] = len(self.dict)

        #翻转词典，使word成为value,index为key,后面可以根据序列得到该词
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len = None):
        """
        把句子转换为序列
        :param sentence:句子 【w1,w2,w3...】
        :param max_len: 保证长度一致，多则裁，少则填
        :return:
        """
        if max_len is not None and isinstance(max_len,int):
            if max_len>len(sentence):
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
            else:
                sentence = sentence[:max_len]

        return [self.dict.get(word,self.UNK) for word in sentence]

    def inverse_transform(self,indices):
        """
        序列转换为句子
        :param indices:
        :return:
        """
        return [self.inverse_dict.get(index) for index in indices]

    def __len__(self):
        return len(self.dict.items())

if __name__ == '__main__':
    word2seq = Word2Sequence()
    word2seq.fit(["我","是","谁"])
    word2seq.fit(["我", "是", "我"])
    word2seq.build_vocab()
    print(word2seq.dict)
    ret = word2seq.transform(["我","爱","南昌"],max_len=10)
    print(ret)
    print(word2seq.inverse_transform(ret))
    print(word2seq.to_word(2))
    print(word2seq.to_index("你好"))