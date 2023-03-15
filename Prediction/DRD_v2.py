# -*- coding: utf-8 -*-
# 模型预测

import pandas as pd
import numpy as np
import datetime
import os
import warnings
import re
import cn2an
from itertools import combinations
import matplotlib.pyplot as plt
import re
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''文件读取'''

path = os.path.dirname(os.path.abspath('file')).replace('\\','/',100) 
def Check_df(path,nam):

    load = path + "/" + nam

    for i in ('utf-8','gb18030'): 
        try:

            data = pd.read_csv( load ,encoding = i)
            print('读取' + nam)
            return data 
        except:
            pass

def read_excel_with_file_name(file_name):
    with warnings.catch_warnings(record=True):
        file_path = path + "/" + file_name
        wb = pd.read_excel(file_path)
        print('读取' + file_name)
    return wb   

'''文档预处理：人名处理（X，数字），引索设置、、、'''


# Excel原数据读取：
LegalDoc = read_excel_with_file_name('_smgtxt.xlsx')


# 替换X和数字：
LegalDoc['def_NoX'] = LegalDoc['def'] # 用于去除X和数字
LegalDoc['def_NoX'] = LegalDoc['def_NoX'].str.replace('X','某') # 替换X
LegalDoc['def_NoX'] = LegalDoc['def_NoX'].str.replace('×','某') # 替换X
LegalDoc['def_NoX'] = LegalDoc['def_NoX'].str.replace('Ｘ','某') # 替换X
LegalDoc['def_NoX'] = LegalDoc['def_NoX'].str.replace('x','某') # 替换X

LegalDoc_1 = LegalDoc

# LegalDoc_1= LegalDoc.replace({'def_NoX':['X','×','Ｘ','x']},'某') # 替换X issue: 无法替换英文字母X
for i in LegalDoc_1['def_NoX']:
    j = cn2an.transform(i,"an2cn") # 替换数字
    LegalDoc_1['def_NoX'] = LegalDoc_1['def_NoX'].replace([i], j)

# 利用人名字典替换
## notice: 若文本中带数字人名没有出现在被告人名列中（即 没有文本中此被告人）该人名数字将不会被替换


name_dict = LegalDoc_1.set_index(['def'])['def_NoX'].to_dict() # 生成替换字典
name_dict_sorted = dict(sorted(name_dict.items(), key=lambda e: e[1], reverse= True)) # 人名长度倒序，排除人名前两个字同名的情况
LegalDoc_1["name_transformed"] = None
def rep(rawstr, dict_rep):
    for i in dict_rep:
        rawstr = rawstr.replace(i, dict_rep[i])
    return rawstr

for j in range(0,len(LegalDoc_1)):
    
    a = rep(LegalDoc_1['_ft'].iloc[j], name_dict_sorted)
    LegalDoc_1["name_transformed"].iloc[j] = a # 生成的替换列“name_transformed”


'''设置index'''
# groupby 后删除组内的重复数据:以人物Groupby，得到身份标注后，关联案件。
# LegalCase = LegalDoc[['index', 'name_transformed']]
Case_df_1 = LegalDoc_1.groupby(['case','def_NoX']).name_transformed.unique() # 以人名为单位
Case_df_1_df = pd.DataFrame(Case_df_1)

'''筛选案件人数1(2)人以上的案件'''

def case_sep(dataset):

    Case_Nm = list(dataset.index.levels[0])

    single_def = []
    multi_def = []

    for name in Case_Nm:

        def_name = dataset.loc[name]

        if len(def_name) < 2:

            single_def.append(name)

        else:

            pass

    single_def_out = list(single_def)

    return single_def_out

'''separate the case file'''

single_def = case_sep(Case_df_1_df)
case_list = list(Case_df_1_df.index.levels[0])
multi_def = [i for i in case_list if i not in single_def]

multi_def_df = Case_df_1_df.loc[multi_def] # 多被告案件
single_def_df = Case_df_1_df.loc[single_def] # 单被告案件
multi_def_df = multi_def_df.reset_index()
multi_def_df = multi_def_df.set_index(['case', 'def_NoX']) #重设index

'''Case info combination'''
# 案件组合模块：

def com_name(dataset):

    Case_No = list(dataset.index.levels[0])

    Case_data = pd.DataFrame()

    Comb_count = []

    for case in Case_No:


        case_info = dataset.loc[case]
        name_list = list(case_info.index)
        name_comb = list(combinations(name_list,2)) # 人名排列组合
        head = len(name_list) - 1 # 只取第一个被告与其他被告的组合
        name_comb_head = name_comb[:head] # 只取第一个被告与其他被告的组合
        name_comb_dict = {'p':name_comb_head}
        name_comb_df = pd.DataFrame(name_comb_dict)
        name_comb_df[['人物1', '人物2']] = name_comb_df['p'].apply(pd.Series)
        name_df = name_comb_df.drop(['p'], axis=1) # 生成案件内人物排列组合表
        df = pd.DataFrame(columns=['case','人物1', '人物2', 'text'])
        # print(len(name_df))
        Comb_count.append(len(name_list))
        
        
        
        for i in range(len(name_df)): # 每一组匹配句子：
                                                            #<---bug
            # print(i) # 组内循环次数 
            text = []
            for j in name_df.loc[i]:  # 遍历组内人物：

                # p1 p2

                def_text = case_info.loc[j]
                # print("def_text:",def_text) # <-- total info of each df

                def_text_list = list(def_text['name_transformed'])

                # print("def_text_list:",def_text_list) # <-- text of each df
                for tx in def_text_list:
                    if not tx in text:
                        text.append(tx) # <-- drop duplication while combining sentence
                # print("text:", text) # <-- both two df's text completed here

            text_new = [str(x) for x in text] 
            text_str = ",".join(text_new)

    

            # name_df['text'].loc[i] = text
            df.loc[i,'text'] = text_str # <-- write in to df with each Case with name and case_text combined
            # print(df)
                # print(def_text_list)
        df['人物1'] = name_df['人物1']
        df['人物2'] = name_df['人物2']
        df['case'] = case
        # print("Final_text:  ",df)
    
    
        Case_data = Case_data.append(df) 

    return Case_data,Comb_count

# if __name__ == '__main__':
 
#     Case_info_trans_test, Case_com_count_test = com_name(test_df)

'''去除模糊案件'''
# 案件内容包含主从犯字眼等

# 合并案件文本：
def listToStr(dataset):
    bb = dataset
    aa = bb["name_transformed"] 
    for i in range(len(aa)):
        Case_info = aa.loc[i]
        Case_df = ",".join(Case_info)
        bb.loc[i,'name_transformed'] = Case_df
    
    return bb

sele_sen = multi_def_df.reset_index() # 重置index
sele_sen = listToStr(sele_sen)

sele_sen_df = sele_sen.groupby(['case']).name_transformed.unique() # 按案件合并文本，并去重
sele_sen_df_Noindex = sele_sen_df.reset_index()
sele_sen_df_Noindex = listToStr(sele_sen_df_Noindex)

case_name_list = [] #含主或从犯字眼的案件名称
# case_name_list_2 = []

for i in range(len(sele_sen_df_Noindex)):
    if ("主犯" in sele_sen_df_Noindex.loc[i,'name_transformed']) or ("从犯" in sele_sen_df_Noindex.loc[i,'name_transformed']): # 'and'作为筛选条件的话会把双主犯的案件筛掉
        # print(sele_sen_df_Noindex.loc[i,'name_transformed'])
        # case_name_list.append(sele_sen_df_Noindex.loc[i,'case'])
        case_name_list.append(sele_sen_df_Noindex.loc[i,'case'])

# 输出训练集：
# train_case_0131 = multi_def_df.loc[case_name_list]
multi_def_df_total = multi_def_df.loc[case_name_list]

'''预测数据输入'''
# 预测数据准备

multi_def_df_total_input = multi_def_df_total.reset_index()
multi_def_df_total_input = multi_def_df_total_input.set_index(['case', 'def_NoX']) #重设index
# 人物关系组合（只取第一个被告与其他被告的组合）：
Case_Pre_df, Case_Pre_Case_com_count = com_name(multi_def_df_total_input)

# 待预测数据：
Case_Pre_df_trans = Case_Pre_df.reset_index(drop=True)
Case_Pre_df_trans = Case_Pre_df_trans.groupby('case').apply(lambda x:x[:]).drop(axis=1, columns='case', inplace=False) # 按案件名称分组

# Case_Pre_df_trans['人物对'] = Case_Pre_df_trans["人物1"] + "-" + Case_Pre_df_trans["人物2"]
# Case_Pre_df_trans = Case_Pre_df_trans.reset_index()

'''文本输入预测'''
# run in "people" evn

import os, json
import numpy as np
from bert.extract_feature import BertVector
from keras.models import load_model
from att import Attention
from tqdm import tqdm
from keras import backend as K
import time
import tensorflow as tf
import gc




# 加载训练效果最好的模型
model_dir = './models'
files = os.listdir(model_dir)
models_path = [os.path.join(model_dir, _) for _ in files]
best_model_path = sorted(models_path, key=lambda x: float(x.split('-')[-1].replace('.h5', '')), reverse=True)[0]
# print(best_model_path)
# K.clear_session()
# model = load_model(best_model_path, custom_objects={"Attention": Attention})
# graph = tf.get_default_graph()


Case_Pre_df_trans_after = Case_Pre_df_trans
Case_Pre_df_trans_after['关系'] = ''

# 读取预测数据：
length = len(Case_Pre_df_trans_after)

for i in tqdm(range(length)):
    
    
    K.clear_session() #在调用BertVector之前，清空一下缓存

    model = load_model(best_model_path, custom_objects={"Attention": Attention}) # 每次预测完成之后，清空session之后，之后重新加载模型
    # graph = tf.get_default_graph()

    case_info = Case_Pre_df_trans_after.iloc[i]

    text1 = case_info['人物1'] + '#' + case_info['人物2'] + '#' + case_info['text']

    per1, per2, doc = text1.split('#')
    text = '$'.join([per1, per2, doc.replace(per1, len(per1)*'#').replace(per2, len(per2)*'#')])
    # print(text)


    # 利用BERT提取句子特征

    bert_model = BertVector(pooling_strategy="NONE", max_seq_len=256)
    vec = bert_model.encode([text])["encodes"][0]
    x_train = np.array([vec])

    # 模型预测并输出预测结果
    # with graph.as_default():
    predicted = model.predict(x_train)
     
    gc.collect()
    y = np.argmax(predicted[0])
    # tf.get_default_graph().finalize()

    with open('data/rel_dict.json', 'r', encoding='utf-8') as f:
        rel_dict = json.load(f)

    id_rel_dict = {v:k for k,v in rel_dict.items()}
    print('原文: %s' % text1)
    print('预测人物关系: %s' % id_rel_dict[y])

    case_info['关系'] = id_rel_dict[y]
    Case_Pre_df_trans_after.iloc[i] = case_info



Case_Pre_df_trans_after.to_excel('关系输出.xls')
