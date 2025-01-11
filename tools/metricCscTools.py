import os
def metricThroughtTsvFile(test_input_id_list, test_input_sentence_list, test_output_list, test_dataset_tsv_path,test_num):
    '''
    params:
        
        test_input_sentence_list: 查询语句的列表
        test_input_id_list: 查询语句的id列表
        test_output_list: 预测结果列表
        test_dataset_tsv_path: 测试集答案的tsv文件路径
        test_num: 进行测试语句数目
    '''
    current_directory = os.getcwd()
    # 要删除的文件路径
    predtuple_path = os.path.join(current_directory, "predtuple.txt")
    answertuple_path = os.path.join(current_directory, "answertuple.txt")

    # 检查并删除文件
    if os.path.exists(predtuple_path): os.remove(predtuple_path)
    if os.path.exists(answertuple_path): os.remove(answertuple_path)
    # generate tuple txt from predStrList 
    predTxtPath = os.path.join(current_directory, "predtuple.txt")
    

    getPredTupleFromSentences(predTxtPath,test_output_list,test_input_sentence_list,test_input_id_list)
    # generate tuple txt from pathOfTestAnswerFile
    answerTxtPath = os.path.join(current_directory,"answertuple.txt")
    getSpecifyNumDataFromTsv(test_dataset_tsv_path,answerTxtPath, test_num)
    # metric and get score 
    metricAnalysisSavePath = os.path.join(current_directory, "resultMetricAnalysis.txt")
    score = metric(predTxtPath,answerTxtPath,metricAnalysisSavePath)
    return score 



def getPredTupleFromSentences(tupletxtPath, predStrList, querySentencesList, querySentencesIdList):
    '''
    把模型预测后的句子列表处理成可以评估精确度的txt格式文件
    :tupletxtPath: 处理后的文件地址
    :predStrList: 模型预测结果列表
    :querySentencesList: 查询句子列表
    :querySentencesIdList: 查询句子的id列表
    '''
    predTupleList = []
    for i in range(len(predStrList)):
        pred_lbl = tupleLabelFormat(predStrList[i],querySentencesList[i],querySentencesIdList[i])
        predTupleList.append(pred_lbl)
    # 写入信息
    with open(tupletxtPath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(predTupleList))
    print(f'\n\nMetric write to "{tupletxtPath}"')
    f.close()

def getSpecifyNumDataFromTsv(answerTsvPath,answerTuplePath,num):
    '''
    从测试答案的tsv文件中取得指定数目的数据，存放到txt中方便与预测数据进行比较
    :answerTsvPath: 存放正确结果tsv的位置
    :answerTuplePath: 将要转换成的txt的位置
    :num: 获取的数目
    '''

    with open(answerTsvPath, 'r',encoding='utf-8') as tsv_file:
        lines = tsv_file.readlines()


    if num<=0 : num = len(lines)
    with open(answerTuplePath, 'w',encoding='utf-8') as txt_file:
        for i,line in enumerate(lines):
            if(i >= num):
                break
            txt_file.write(line.replace('\t', ', '))
    print(f'\n\nMetric write to "{answerTuplePath}"')
    tsv_file.close()
    txt_file.close()



def metric(predPath, answerPath,metricAnalysisSavePath):
    '''
    评测预测结果所得分数
    :predPath: 预测结果所在路径
    :answerPath: 正确结果所在路径
    '''
    preds = read_file(predPath)
    targs = read_file(answerPath)
    results = {}
    res,detect_hit_list,detect_not_hit_list,detect_tpList = sent_metric_detect(preds=preds, targs=targs)
    results.update(res)
    res,correct_hit_list,correct_not_hit_list,correct_tpList = sent_metric_correct(preds=preds, targs=targs)
    results.update(res)
    metricAnalysisSave(detect_hit_list,detect_not_hit_list,detect_tpList,correct_hit_list,correct_not_hit_list,correct_tpList,metricAnalysisSavePath)
    return results


def metricAnalysisSave(detect_hit_list,detect_not_hit_list,detect_tpList,correct_hit_list,correct_not_hit_list,correct_tpList,metricAnalysisSavePath):
    '''
    :detect_hit_list: 检测结果和正确答案一致的id列表
    :detect_not_hit_list: 检测结果和正确答案不一致的id列表
    :detect_tpList: 检测结果和正确答案一致，并且实际存在错误字符的id列表
    :correct_hit_list: 更改后结果和正确答案一致的id列表
    :correct_not_hit_list: 更改后结果和正确答案不一致的id列表
    :correct_tpList: 更改后的结果和正确答案一致，并且实际存在错误字符的id列表
    分析并保存一些测试结果
    '''
    with open(metricAnalysisSavePath,'w')as file:
        file.write(f"检测结果和正确答案一致的id列表:{detect_hit_list}\n")
        file.write(f"检测结果和正确答案不一致的id列表:{detect_not_hit_list}\n")
        file.write(f"检测结果和正确答案一致，并且实际存在错误字符的id列表:{detect_tpList}\n")
        file.write(f"更改后结果和正确答案一致的id列表:{correct_hit_list}\n")
        file.write(f"更改后结果和正确答案不一致的id列表:{correct_not_hit_list}\n")
        file.write(f"更改后的结果和正确答案一致，并且实际存在错误字符的id列表:{correct_tpList}\n")
    file.close()


def tupleLabelFormat(predStr, queryStr,strIds):
    '''
    比较predStr,queryStr,将其中不同的地方找到并且生成指定格式的字符串返回
    :predStr: 预测句子
    :queryStr: 待查询句子
    :strIds: 当前句子id
    '''
    # 调整长度
    min_value = min(len(predStr),len(queryStr))
    predStr = predStr[:min_value]
    queryStr = queryStr[:min_value]
    # 转为lbl同格式
    item = []
    item.append(str(strIds))
    for i, (a, b) in enumerate(zip(queryStr, predStr), start=1):
        if a != b:
            item.append(str(i))
            item.append(str(b))
    if len(item) == 1:
        item.append('0')
    pred_lbl = ', '.join(item)
    return pred_lbl



def read_file(path):
    '''
    读取预测文件和正确文件
    :path: 文件路径
    item = [id,(num,cha),(num,cha),(num,cha)]
    '''
    with open(path, 'r', encoding='utf-8') as f:
        rows = [r.strip().split(', ') for r in f.read().splitlines()]

    data = []
    for row in rows:
        item = [row[0]]
        if(len(row)<2): continue
        if  (len(row[0])<5): continue
        data.append(item)
        if len(row) == 2 and row[1] == '0':
            continue
        for i in range(1, len(row), 2):
            if(i + 1 <len(row)):
                item.append((int(row[i]), row[i + 1]))
    return data



def sent_metric_detect(preds, targs):
    '''
    获得检测正确率
    :preds:是预测结果
    :targs:是正确结果
    A2-0085-1, 8, 勤, 9, 奋
    '''
    assert len(preds) == len(targs)
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    hit_list = []
    not_hit_list = []
    tpList = []
    for pred_item, targ_item in zip(preds, targs):
        try:
            pred_item[0] == targ_item[0] # 比较id
        except Exception as e:
            print(e)
            continue
            
        pred, targ = sorted(pred_item[1:]), sorted(targ_item[1:]) # 比较预测的错字
        if targ != []:
            targ_p += 1
        if pred != []:
            pred_p += 1
        if len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)):
            hit += 1
            hit_list.append(pred_item[0])
        else:
            not_hit_list.append(pred_item[0])
        if pred != [] and len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)):
            tp += 1
            tpList.append(pred_item[0])

    acc = hit / len(targs)
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    print(f"tp是{tp},pred_p是{pred_p},targ_p是{targ_p},hit是{hit}")
    results = {
        'sent-detect-acc': acc * 100,
        'sent-detect-p': p * 100,
        'sent-detect-r': r * 100,
        'sent-detect-f1': f1 * 100,
    }
    formatted_results = format_results(results)  
    return formatted_results,hit_list,not_hit_list,tpList


def sent_metric_correct(preds, targs):
    '''
    获得纠错正确率
    :preds:是预测结果
    :targs:是正确结果
    '''
    hit_list = []
    not_hit_list = []
    tpList = []
    assert len(preds) == len(targs)
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for pred_item, targ_item in zip(preds, targs):
        try:
            pred_item[0] == targ_item[0] # 比较id
        except Exception as e:
            print(e)
            continue
        pred, targ = sorted(pred_item[1:]), sorted(targ_item[1:])
        if targ != []:
            targ_p += 1
        if pred != []:
            pred_p += 1
        if pred == targ:
            hit += 1
            hit_list.append(pred_item[0])
        else:
            not_hit_list.append(pred_item[0])
        if pred != [] and pred == targ:
            tp += 1
            tpList.append(pred_item[0])

    acc = hit / len(targs)
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = {
        'sent-correct-acc': acc * 100,
        'sent-correct-p': p * 100,
        'sent-correct-r': r * 100,
        'sent-correct-f1': f1 * 100,
    }
    formatted_results = format_results(results)
    return formatted_results,hit_list,not_hit_list,tpList


def format_results(results):
    '''
    对结果进行格式化
    :results: metric 的结果
    '''
    formatted_results = {}
    for key, value in results.items():
        formatted_results[key] = round(value, 2)
    return formatted_results