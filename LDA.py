
import re
import jieba
import chardet
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

def choose_topic(corpus, dic):
    '''
    @description: 生成模型
    @param
    @return: 生成主题数分别为1-5的LDA主题模型，并保存起来。
    '''
    for i in range(1, 6):
        print('目前的topic个数:{}'.format(i))
        temp = 'lda_{}'.format(i)
        tmp = LdaModel(corpus, num_topics=i, id2word=dic, passes=20)
        topics = tmp.print_topics(num_words=10)
        for topic in topics:
            print(topic)

        topic_keywords = []
        for topic_id in range(i):
            topic_terms = tmp.get_topic_terms(topic_id, topn=10)
            topic_words = [dic.get(term[0]) for term in topic_terms]
            topic_keywords.append(' '.join(topic_words))
            print(topic_keywords[-1])

        # 生成词云
        font = r'D:\download\chrome\SourceHanSerifSC-VF.ttf'
        wordcloud = WordCloud(collocations=False, font_path=font, width=1400, height=1400, margin=2)
        wordcloud.generate_from_text(topic_keywords)

        # 显示词云
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

        # 保存词云到文件
        image_path = './{}_wordcloud.png'.format(temp)
        wordcloud.to_file(image_path)

        # 保存模型
        file_path = './{}.model'.format(temp)
        tmp.save(file_path)
        print('------------------')

def visible_model(topic_num, corpus):
    '''
    @description: 可视化模型
    @param :topic_num:主题的数量
    @return: 可视化lda模型
    '''
    x_list = []
    y_list = []
    for i in range(1, topic_num):
        temp_model = 'lda_{}.model'.format(i)
        try:
            lda = models.ldamodel.LdaModel.load(temp_model)
            cv_tmp = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherence_score = cv_tmp.get_coherence()
            x_list.append(i)
            y_list.append(coherence_score)
        except:
            print('没有这个模型:{}'.format(temp_model))

    plt.plot(x_list, y_list)
    plt.xlabel('num topics')
    plt.ylabel('coherence score')
    plt.legend(('coherence_values'), loc='best')
    plt.show()

if __name__ == "__main__":
    file_path1 = 'text_to_sentence_no.csv'
    detect_encoding(file_path1)
    df = pd.read_csv(file_path1, encoding='gbk')
    df['cut'] = df['sentence'].apply(preprocess_text)
    texts = [doc.split() for doc in df['cut']]
    dictionary = corpora.Dictionary(texts)
    doc_term_matrix = [dictionary.doc2bow(text) for text in texts]

    choose_topic(doc_term_matrix, dictionary)
    visible_model(6, doc_term_matrix)
