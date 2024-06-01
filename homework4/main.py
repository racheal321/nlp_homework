import gensim
from gensim.models import Word2Vec
import jieba


# 1. 加载和准备语料库
def read_cn_stopwords():
    cn_stopwords = []
    with open('D:/zongruntang/pythonProject2/jyxstxtqj_downcc.com/cn_stopwords.txt', mode='r', encoding='utf-8') as f:
        cn_stopwords.extend([line.strip() for line in f.readlines()])
    print('成功加载停词')
    return cn_stopwords


def read_novel(path):
    para_list = []

    # 读取小说列表
    with open(file=path + 'inf.txt', mode='r', encoding='gb18030') as f:
        file_names = f.readline().split(',')

    # 读取小说，去掉广告，去掉较短的句子
    for index, name in enumerate(file_names):
        novel_name = path + name + '.txt'
        with open(file=novel_name, mode='r', encoding='gb18030') as f:
            content = f.read()
            ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'
            content = content.replace(ad, '')
            for sentence in content.split('\n'):
                if len(sentence) < 500:
                    continue
                para_list.append(sentence)

    stop_words = read_cn_stopwords()
    data = []
    for paragraph in para_list:
        fenci = jieba.cut(paragraph)
        filtered_words = [word for word in fenci if word.strip() and word not in stop_words]
        data.append(filtered_words)  # 使用append而不是extend

    print("成功分词")
    return data


# 2. 训练Word2Vec模型
def train_word2vec(data):
    model = Word2Vec(sentences=data, vector_size=200, window=20, min_count=10, workers=7,epochs=50)
    print("完成训练")
    return model


# 3. 验证词向量的准确性
def test_word2vec(model, words):
    for word in words:
        # 找到一个词的最近邻
        if word in model.wv:
            similar_words = model.wv.most_similar(word)
            print(f"与 '{word}' 最相似的词:")
            for similar_word, similarity in similar_words:
                print(f"{similar_word}: {similarity:.4f}")
        else:
            print(f"词 '{word}' 不在词汇表中。")

    # 计算两个词的相似度
    word_pairs = [("武功", "内力"), ("丐帮", "魔教")]  # 替换成你想比较的词对
    for word1, word2 in word_pairs:
        if word1 in model.wv and word2 in model.wv:
            similarity = model.wv.similarity(word1, word2)
            print(f"'{word1}' 和 '{word2}' 的相似度: {similarity:.4f}")
        else:
            print(f"词 '{word1}' 或 '{word2}' 不在词汇表中。")


if __name__ == "__main__":
    # 替换成你的语料库文件路径
    corpus_path = "D:/zongruntang/pythonProject2/jyxstxtqj_downcc.com/"

    # 读取和处理语料库
    data = read_novel(corpus_path)

    # 训练Word2Vec模型
    model = train_word2vec(data)

    # 验证词向量的准确性

    words_to_test = ["丐帮", "内功", "魔教", "赵敏"]
    test_word2vec(model, words_to_test)

