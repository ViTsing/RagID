from transformers import BertTokenizer
import pickle
import copy
import torch

def processInit():
    # tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


    chaizi=pickle.load(open('./distribution/character_feature/chaizi.pkl','rb'))
    convert=pickle.load(open('./distribution/character_feature/convert.pkl','rb'))
    finals=pickle.load(open('./distribution/character_feature/finals.pkl','rb'))
    initials=pickle.load(open('./distribution/character_feature/initials.pkl','rb'))
    structure=pickle.load(open('./distribution/character_feature/structure.pkl','rb'))
    tone=pickle.load(open('./distribution/character_feature/tone.pkl','rb'))
    write_num=pickle.load(open('./distribution/character_feature/write_num.pkl','rb'))

    finals['ang']=finals['ing']+finals.pop('an')
    finals['eng']=finals['ing']+finals.pop('en')
    finals['ing']=finals['ing']+finals.pop('in')



    chaizi_dict=dict()
    for item in chaizi:
        lst=tokenizer.convert_tokens_to_ids(item[1])
        while 100 in lst:
            lst.remove(100)
        if len(lst):
            chaizi_dict[tokenizer.convert_tokens_to_ids(item[0])]=lst

    pickle.dump(chaizi_dict,open('./distribution/token_replace_dict/chaizi_dict.pkl','wb'))

    same_pianpang=dict()
    for key,value in chaizi_dict.items():
        for item in value:
            if same_pianpang.get(item,None):
                if key not in same_pianpang[item]:
                    same_pianpang[item].append(key)
            else:
                same_pianpang[item]=[key]

    pianpang_dict=dict()
    for values in same_pianpang.values(): 
        lst = values
        for item in lst:
            tem_lst=copy.deepcopy(lst)
            tem_lst.remove(item)
            if len(tem_lst):
                if pianpang_dict.get(item,None):
                    pianpang_dict[item]=list(set(pianpang_dict[item]+tem_lst))
                else:
                    pianpang_dict[item]=tem_lst

    pickle.dump(pianpang_dict,open('./distribution/token_replace_dict/pianpang_dict.pkl','wb'))




    convert_dict=dict()
    for item in convert:
        char_1=tokenizer.convert_tokens_to_ids(item[0])
        if convert_dict.get(char_1, None):
            convert_dict[char_1].append(tokenizer.convert_tokens_to_ids(item[1]))
        else:
            convert_dict[char_1]=[tokenizer.convert_tokens_to_ids(item[1])]
    pickle.dump(convert_dict,open('./distribution/token_replace_dict/convert_dict.pkl','wb'))


    finals_dict=dict()
    for values in finals.values(): 
        lst=tokenizer.convert_tokens_to_ids(values)
        while 100 in lst:
            lst.remove(100)
        for item in lst:
            tem_lst=copy.deepcopy(lst)
            tem_lst.remove(item)
            if len(tem_lst):
                finals_dict[item]=tem_lst
    pickle.dump(finals_dict,open('./distribution/token_replace_dict/finals_dict.pkl','wb'))



    initials_dict=dict()
    for values in initials.values(): 
        lst=tokenizer.convert_tokens_to_ids(values)
        while 100 in lst:
            lst.remove(100)
        for item in lst:
            tem_lst=copy.deepcopy(lst)
            tem_lst.remove(item)
            if len(tem_lst):
                initials_dict[item]=tem_lst
    pickle.dump(initials_dict,open('./distribution/token_replace_dict/initials_dict.pkl','wb'))



    structure_dict=dict()
    for values in structure.values(): 
        lst=tokenizer.convert_tokens_to_ids(values)
        while 100 in lst:
            lst.remove(100)
        for item in lst:
            tem_lst=copy.deepcopy(lst)
            tem_lst.remove(item)
            if len(tem_lst):
                structure_dict[item]=tem_lst
    pickle.dump(structure_dict,open('./distribution/token_replace_dict/structure_dict.pkl','wb'))


    tone_dict=dict()
    for values in tone.values(): 
        lst=tokenizer.convert_tokens_to_ids(values)
        while 100 in lst:
            lst.remove(100)
        for item in lst:
            tem_lst=copy.deepcopy(lst)
            tem_lst.remove(item)
            if len(tem_lst):
                tone_dict[item]=tem_lst
    pickle.dump(tone_dict,open('./distribution/token_replace_dict/tone_dict.pkl','wb'))




    write_num_dict=dict()
    for values in write_num.values(): 
        lst=tokenizer.convert_tokens_to_ids(values)
        while 100 in lst:
            lst.remove(100)
        for item in lst:
            tem_lst=copy.deepcopy(lst)
            tem_lst.remove(item)
            if len(tem_lst):
                write_num_dict[item]=tem_lst
    pickle.dump(write_num_dict,open('./distribution/token_replace_dict/write_num_dict.pkl','wb'))

    with open("./distribution/token_replace_dict/flag.txt", 'w') as file:
        file.write("1")
    file.close()