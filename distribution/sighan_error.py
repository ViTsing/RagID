from .hanzi_chaizi import HanziChaizi
import os
import pdb
import pickle
import pandas as pd
from transformers import BertTokenizer
import pickle
import copy
import torch
from pypinyin import pinyin, lazy_pinyin, Style

# pdb.set_trace()
# INITIALS,FINALS

hc = HanziChaizi()

def calculateDistribution(output_file):
    path=[]

    error_stat={
        'set':[],
        'p_error':[],
        'g_error':[],
        'pg_error':[],
        'others':[],
    }
    for path in os.listdir('distribution/data'):
        # data = pd.read_csv(os.path.join('data',path),sep='\t',header=None)
        data = pd.read_csv(os.path.join('distribution/data', path), sep=',', header=0)
        # Check the column name of the second column
        if data.columns[1] == 'label':
            # If the second column is 'label', delete the second column
            data = data.drop(columns=[data.columns[1]])

        # tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        with open('./distribution/token_replace_dict/chaizi_dict.pkl',"rb") as f:
            chaizi_dict=pickle.load(f)

        with open('./distribution/token_replace_dict/pianpang_dict.pkl',"rb") as f:
            pianpang_dict=pickle.load(f)  

        p_error=0
        g_error=0
        pg_error=0
        others=0
        error_ana=[]

        for item in data.iterrows():
            for i in range(len(item[1][1])):
                if item[1][1][i]!=item[1][2][i] and '\u4e00' <= item[1][1][i] <= '\u9fff' and '\u4e00' <= item[1][2][i] <= '\u9fff':
                    try:
                        glyph_flag=list((set(hc.query(item[1][1][i])) & set(hc.query(item[1][2][i]))))!=[]
                    except:
                        try:
                            glyph_flag=item[1][2][i] in hc.query(item[1][1][i])
                        except:
                            try:
                                glyph_flag=item[1][1][i] in hc.query(item[1][2][i])
                            except:
                                glyph_flag=False
                    pron_flag=judge_pron(item[1][1][i],item[1][2][i])



                    if glyph_flag and pron_flag:
                        error_ana.append((item[1][1][i],item[1][2][i],'both'))
                        pg_error+=1
                    elif glyph_flag:
                        error_ana.append((item[1][1][i],item[1][2][i],'glyph'))
                        g_error+=1
                    elif pron_flag:
                        error_ana.append((item[1][1][i],item[1][2][i],'pron'))
                        p_error+=1       
                    else:
                        error_ana.append((item[1][1][i],item[1][2][i],'others'))
                        others+=1         


        error_stat['set'].append(path)
        error_stat['p_error'].append(p_error)
        error_stat['g_error'].append(g_error)
        error_stat['pg_error'].append(pg_error)
        error_stat['others'].append(others)


        results=pd.DataFrame(error_stat)
        results.to_csv(output_file,mode='a',index=False)



def judge_glyph(query,target):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    query_id = tokenizer.convert_tokens_to_ids(query)
    target_id = tokenizer.convert_tokens_to_ids(target)
    with open('./token_replace_dict/chaizi_dict.pkl',"rb") as f:
        chaizi_dict=pickle.load(f)
    if query_id in chaizi_dict[target_id] or target_id in chaizi_dict[query_id] or query_id in pianpang_dict[target_id] or target_id in pianpang_dict[query_id]:
        return True
    else:
        return False
    
def judge_pron(query,target):
    query_initials=pinyin(query, style=Style.INITIALS,strict=False,neutral_tone_with_five=True)[0][0]
    query_finals=pinyin(query, style=Style.FINALS,strict=False,neutral_tone_with_five=True)[0][0]
    target_initials=pinyin(target, style=Style.INITIALS,strict=False,neutral_tone_with_five=True)[0][0]
    target_finals=pinyin(target, style=Style.FINALS,strict=False,neutral_tone_with_five=True)[0][0]
    similar_initials=[
        ['z','zh'],
        ['s','sh'],
        ['c','ch'],
        ['n','l']
    ]
    similar_finals=[
        ['a','ia','ua'],
        ['o','uo'],
        ['ie','ve'],
        ['ai','uai'],
        ['ei','ui'],
        ['ao','iao'],
        ['ou','iu'],
        ['an','uan','ian'],
        ['en','in','un'],
        ['ang','iang','uang'],
        ['eng','ing'],
        ['ong','iong'],
        ['an','ang'],
        ['en','eng'],
        ['in','ing']
        ]
    if query_initials==target_initials:
        if query_finals==target_finals:
            return True
        flag=False
        for finals in similar_finals:
            if query_finals in finals and target_finals in finals:
                flag = True
                break
        return flag
    else:
        if query_finals==target_finals:
            if query_initials=='' or target_initials=='':
                return True
            flag=False
            for initials in similar_initials:
                if query_initials in initials and target_initials in initials:
                    flag = True
                    break
            return flag
        else:
            return False
