# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
from sentence_transformers import SentenceTransformer
#import MeCab
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

#def tokenizer(text):
#  wakati = MeCab.Tagger('-Owakati')
#  return wakati.parse(text).strip('\n')

#top_key = st.text_input('输入top数')


def find_most_similar_words(embedding_list, target_embedding, top_k=1):
    similarity_scores = cosine_similarity([target_embedding], embedding_list)[0]
    most_similar_indices = np.argsort(similarity_scores)[::-1][:top_k]
    most_similar_scores = similarity_scores[most_similar_indices]

    return most_similar_indices, most_similar_scores


# Assuming `embedding_list` is a list of word embeddings
# Assuming `target_embedding` is the embedding of the target word
#start_time = time.time()
#target_word = tokenizer('キーホルダー')
#target_word_embedding = model.encode(target_word)
# target_word_embedding = get_embedding(target_word)  # Get the embedding of the target word
#similar_indices, similarity_scores = find_most_similar_words(embedding_list, target_word_embedding)

##for index, score in zip(similar_indices, similarity_scores):
#    word = embedding_list[index]
#    print(
#        f"{data['Category_collection'][index]}, {data['cat_name'][index]}, {data['cat_code'][index]}, {data['all_catl'][index]}, score: {score}")
#    # print(data.iloc[index,:])
#end_time = time.time()
#print('processing time is {}s'.format(end_time - start_time))
embedding_file_ja = './embeddings_cat_ja.npy'
embedding_file_ch = './embeddings_cat_ch.npy'
category_file = './category_col.csv'
def main():

    st.title('Text to Category Generator')
    option = st.selectbox(
        "中文或日文",
        ("中文", "日文"),)
       # label_visibility=st.session_state.visibility,
       # disabled=st.session_state.disabled)
    if option == '日文':
        embedding_list = np.load(embedding_file_ja)
        model = SentenceTransformer("./sbert-base-ja")

    elif option == '中文':
        embedding_list = np.load(embedding_file_ch)
        model = SentenceTransformer("./text2vec-base-chinese")
    category_info = pd.read_csv(category_file)

    uploaded_file = st.file_uploader("选择上传文件-文本列请命名为 Text")

    if uploaded_file is not None:
    #if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
    #    dataframe = pd.read_csv(uploaded_file)
    #    st.write(dataframe)
        dataframe = pd.read_csv(uploaded_file)
        df_col = []
        for j in dataframe['Text']:
                dic = {}
                emb = model.encode(j)
                index, score = find_most_similar_words(embedding_list, emb, top_k=1)
                dic['关键词'] = j
                dic['类目日文'] = category_info['Category_collection'][index].tolist()[0]
                dic['类目中文'] = category_info['cat_name'][index].tolist()[0]
                dic['类目code'] = str(category_info['cat_code'][index].tolist()[0])
                dic['类目级别'] = category_info['all_catl'][index].tolist()[0]
                dic['分数'] = score[0]
                df_col.append(dic)
        df = pd.DataFrame.from_dict(df_col)
        file_out = st.text_input('请输入输出文件名')
        df.to_csv('{}.csv'.format(file_out))

            #st.write(dataframe)

    target_word = st.text_input('单一关键词查询输入')
    number = st.number_input('输入数字-返回topN的关联类目',min_value = 1,step = 1)

    #target_word = 'キーホルダー'
    target_word_embedding = model.encode(target_word)
    similar_indices, similarity_scores = find_most_similar_words(embedding_list, target_word_embedding,top_k = number)
    dd =[]
    for index, score in zip(similar_indices, similarity_scores):
        #word = embedding_list[index]
            d = {}
            d['类目日文'] = category_info['Category_collection'][index]
            d['类目中文'] = category_info['cat_name'][index]
            d['类目code'] = str(category_info['cat_code'][index])
            d['类目级别'] = category_info['all_catl'][index]
            d['分数'] = score
            dd.append(d)
    da = pd.DataFrame.from_dict(dd)
    st.write(da)



main()


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
