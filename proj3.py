import gensim
from gensim import corpora
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import jieba
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
import warnings
warnings.filterwarnings('ignore')


def read_cn_stopwords():
    cn_stopwords = []
    with open('D:/zongruntang/nlp/second/jyxstxtqj_downcc.com/cn_stopwords.txt', mode='r', encoding='utf-8') as f:
        cn_stopwords.extend([line.strip() for line in f.readlines()])
    return cn_stopwords


def read_novel(path):
    para_list = []
    para_label = []
    # 读取小说列表
    with open(file=path + 'inf.txt', mode='r', encoding='gb18030') as f:
        file_names = f.readline().split(',')
    f.close()
    # 读取小说 去掉广告 去掉较短的句子 剩下的段落文本放list 生成对应的索引放label
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
                para_label.append(index)

    return para_list, para_label


def para_extract(para, label):
    # 段落抽取 随机选择1000个段落，后面函数进行token处理
    text_ls = []
    text_label = []
    random_indices = random.sample(range(len(para)), 1000)
    text_ls.extend([para[i] for i in random_indices])
    text_label.extend([label[i] for i in random_indices])
    return text_ls, text_label


def split(text_ls, text_label):
    # 分词
    stop_words = read_cn_stopwords()
    tokens_word = []  # 以词为单位
    tokens_word_label = []
    tokens_char = []  # 以字为单位
    tokens_char_label = []
    for i, text in enumerate(text_ls):
        words = [word for word in jieba.lcut(sentence=text) if word not in stop_words]
        tokens_word.append(words)
        tokens_word_label.append(text_label[i])

        temp = []
        for word in words:
            temp.extend([char for char in word])
        tokens_char.append(temp)
        tokens_char_label.append(text_label[i])

    # 词典与向量集
    dic_word = gensim.corpora.Dictionary(tokens_word)
    dic_char = gensim.corpora.Dictionary(tokens_char)

    corp_word = [dic_word.doc2bow(tokens) for tokens in tokens_word] #doc2bow将字符串转化为词袋向量
    corp_char = [dic_char.doc2bow(tokens) for tokens in tokens_char]
    return dic_word, dic_char, corp_word, corp_char, tokens_word, tokens_char,tokens_word_label, tokens_char_label



def train_lda_model(documents, num_topics):

    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    print('Training LDA model')
    # lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=20, workers=4)
    print('Finished training LDA model')

    return lda_model,dictionary,corpus

#返回的结果就是一个二维数组，其中每一行表示一个文档的主题分布
def get_document_topic_distribution(lda_model, documents):
    topic_distributions = []
    for doc in documents:
        bow_vector = lda_model.id2word.doc2bow(doc)
        topic_distribution = lda_model.get_document_topics(bow_vector, minimum_probability=0.0)
        topic_distribution = [score for _, score in topic_distribution]
        topic_distributions.append(topic_distribution)
    return np.array(topic_distributions)

#利用交叉验证给分类器打分
def evaluate_classification_performance(X, y, classifier):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X, y, cv=kf)
    return scores.mean()

if __name__ == "__main__":
    K_values = [20, 100, 500, 1000, 2500]
    num_topics_values = [20, 40, 60, 80]
    mode_list = ['word', 'char']
    acc_list_KW = {20: [], 100: [], 500: [], 1000: [], 2500: []}
    acc_list_TW = {20: [], 40: [], 60: [], 80: []}
    acc_list_KC = {20: [], 100: [], 500: [], 1000: [], 2500: []}
    acc_list_TC = {20: [], 40: [], 60: [], 80: []}
    path = 'D:/zongruntang/nlp/second/jyxstxtqj_downcc.com/'
    para, label = read_novel(path)
    text_ls, text_label = para_extract(para, label)
    dic_word, dic_char, corp_word, corp_char, tokens_word, tokens_char,tokens_word_label, tokens_char_label = split(text_ls, text_label)
    for K in K_values:
        for num_topics in num_topics_values:
            for mode in mode_list:
                print(f"K = {K}, T = {num_topics}, Mode = {mode}")
                if mode == 'word':
                    all_data, all_label = [row[:K] for row in tokens_word], tokens_word_label
                elif mode == 'char':
                    all_data, all_label = [row[:K] for row in tokens_char], tokens_char_label
                lda_model,dictionary,corpus = train_lda_model(all_data, num_topics)
                all_x_lda = get_document_topic_distribution(lda_model, all_data)
                accuracy = evaluate_classification_performance(all_x_lda, all_label, MultinomialNB())
                print("Accuracy:", accuracy)
                print("=" * 50)
                with open("result.txt", "a") as f:
                    f.write(f"K = {K}, T = {num_topics}, Mode = {mode}\n")
                    f.write(f"Accuracy: {accuracy}\n")
                    f.write("=" * 50 + "\n")
                if mode == 'word':
                    acc_list_KW[K].append(accuracy)
                    acc_list_TW[num_topics].append(accuracy)
                if mode == 'char':
                    acc_list_KC[K].append(accuracy)
                    acc_list_TC[num_topics].append(accuracy)





