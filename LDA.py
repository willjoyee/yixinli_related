#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/1 19:55
# @Author : AwetJodie

from gensim.models import CoherenceModel
import pandas as pd
import re
import jieba
import chardet
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import random

from matplotlib import pyplot as plt


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"Detected encoding: {encoding} (confidence: {confidence})")
def preprocess_text(text):
    pattern = u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!\t"@#$%^&*\\-_=+a-zA-Z，。\n《》、？：；“”‘’｛｝【】（）…￥！—┄－]+'
    cut_text = re.sub(pattern, ' ', text)
    cut_text = " ".join(jieba.lcut(cut_text, cut_all=True))
    return cut_text

file_path1 = 'text_to_sentence_no.csv'
detect_encoding(file_path1)
df = pd.read_csv(file_path1, encoding='gbk')
df['cut'] = df['sentence'].apply(preprocess_text)
texts = [doc.split() for doc in df['cut']]
dictionary = corpora.Dictionary(texts)
doc_term_matrix = [dictionary.doc2bow(text) for text in texts]


def choose_topic(corpus, dic):
    '''
    @description: 生成模型
    @param
    @return: 生成主题数分别为1-15的LDA主题模型，并保存起来。
    '''
    for i in range(1, 10):
        print('目前的topic个数:{}'.format(i))
        temp = 'lda_{}'.format(i)
        tmp = LdaModel(doc_term_matrix, num_topics=i, id2word=dic, passes=20)
        file_path = './{}.model'.format(temp)
        tmp.save(file_path)
        print('------------------')


from gensim.models import CoherenceModel
def visible_model(topic_num, corpus):
        '''
        @description: 可视化模型
        @param :topic_num:主题的数量
        @return: 可视化lda模型
        '''
        x_list = []
        y_list = []
        for i in range(1,10):
            temp_model = 'lda_{}.model'.format(i)
            try:
                lda = models.ldamodel.LdaModel.load(temp_model)
                cv_tmp = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary,coherence='u_mass')
                #计算一致性
                x_list.append(i)
                y_list.append(cv_tmp.get_coherence())
            except:
                print('没有这个模型:{}'.format(temp_model))
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(x_list, y_list)
        plt.xlabel('num topics')
        plt.ylabel('coherence score')
        plt.legend(('coherence_values'), loc='best')
        plt.show()



if __name__ == "__main__":
    choose_topic(doc_term_matrix,dictionary)
    visible_model(16, doc_term_matrix)
