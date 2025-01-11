import pickle
import csv
import pandas as pd
import os


def getFormatFewShotCommentResult(commentResult, input, output):
    '''
    解析结果，看看是否符合要求
    符合要求返回True
    不符合要求返回False

纠错后的句子已经不存在字符错误:是
纠错后的句子对比纠错前的句子，更加合理:是
纠错后的句子和纠错前句子长度相等:是
'''
    lines = commentResult.strip().split('\n')
    if commentResult == "" or commentResult is None:
        return False, commentResult

    dontHaveWrongChar = ""
    moreReasonable = ""
    equalLength = ""

    # 遍历每一行，提取关键信息
    for i, line in enumerate(lines):
        if "字符错误:" in line:
            dontHaveWrongChar = line.strip().split(':')[1]
        if "语法逻辑:" in line:
            moreReasonable = line.strip().split(':')[1]
        if "长度相等:" in line:
            equalLength = line.strip().split(':')[1]
            # 如果输入和输出长度相等，则修改 equalLength 为 "是"
            if len(input) == len(output):
                equalLength = "是"
                lines[i] = "纠错后的句子和纠错前句子长度相等:是"  # 更新对应行的内容
            equalLength = lines[i].strip().split(':')[1]

    # 更新 commentResult
    updated_commentResult = "\n".join(lines)
    # 判断是否符合要求
    if ("是" in dontHaveWrongChar) and ("是" in moreReasonable) and ("是" in equalLength):
        return True, updated_commentResult
    else:
        return False, updated_commentResult
# def getFormatFewShotCommentResult(commentResult,input,output):
#     '''
#     解析结果，看看是否符合要求
#     符合要求返回True
#     不符合要求返回False

# 纠错后的句子已经不存在字符错误:是
# 纠错后的句子对比纠错前的句子，更加合理:是
# 纠错后的句子和纠错前句子长度相等:是
# '''   
#     lines = commentResult.strip().split('\n')
#     if commentResult == "" or commentResult == None:
#         return False    
#     # print(commentResult)
#     dontHaveWrongChar = ""
#     moreReasonable = ""
#     equalLength = ""
#     for line in lines:
#         if "字符错误:" in line:
#             dontHaveWrongChar = line.strip().split(':')[1]
#         if "更加合理:" in line:
#             moreReasonable  = line.strip().split(':')[1]
#         if "长度相等:" in line:
#             equalLength  = line.strip().split(':')[1]
#             print(equalLength)
#             print(commentResult)
#             exit()
#             if(len(input) == len(output)):
#                 ""
#     if ("是" in dontHaveWrongChar) and ("是" in moreReasonable) and ("是" in equalLength or len(input) == len(output)):
#         return True,commentResult
#     else:
#         return False,commentResult


def readPklFile(pkl_path):
    '''
    param:
        pkl_path: path of .pkl file 
    function:
        read data from .pkl file
    '''
    # 读取 pkl 文件
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    f.close()
    return data


def getDictDatasetFromPklDataset(pkl_data):
    '''
    param:
        pkl_data: data from function 'readPklFile'
    function:
        Organize the data read from pkl into a dict-type dataset
    '''
    datasetDict = {
        "inputList": [],
        "outputList": [],
        "lengthList": [],
        "idList": [],
        "num": 0,
    }  # dict-type dataset

    count_of_pkl_data = len(pkl_data)
    if count_of_pkl_data == 0:
        return datasetDict

    datasetDict["num"] = count_of_pkl_data
    
    # Using list comprehensions with default values to avoid None
    datasetDict["inputList"] = [item.get("src", "") for item in pkl_data]  # Default to empty string if 'src' is missing
    datasetDict["outputList"] = [item.get("tgt", "") for item in pkl_data]  # Default to empty string if 'tgt' is missing
    datasetDict["lengthList"] = [item.get("lengths", 0) for item in pkl_data]  # Default to 0 if 'lengths' is missing
    datasetDict["idList"] = [item.get("id", "") for item in pkl_data]  # Default to empty string if 'id' is missing
    return datasetDict


def readCSVFile(csv_path):
    '''
    param:
        csv_path: path of .csv file
    function:
        Get the csv header and content data and return
    '''
    header = []
    content = []

    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            # Read the header
            header = next(csv_reader)
            # Read the content
            content = [row for row in csv_reader]
        
        return header, content
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    



def getFilesFromFolder(folder_path):
    '''
    param: 
        folder_path: path of folder 
    function:
        Get all files' paths in a folder (only one level deep)
    return:
        A list of file paths in the folder
    '''
    try:
        # Get all files and subdirectories in the folder
        all_files = os.listdir(folder_path)

        # Filter out subdirectories and return the full path of files
        file_paths = [os.path.join(folder_path, file) for file in all_files if os.path.isfile(os.path.join(folder_path, file))]

        return file_paths
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []




def save_to_csv(header_list, content_list, csv_path):
    '''
    param:
        header_list: csv header list
        content_list: csv column content corresponding to the header, element in list is also a list
    function:
        store data into csv file
    '''
    # convert content_list to DataFrame
    df = pd.DataFrame({header_list[i]: content_list[i] for i in range(len(header_list))})
    
    # save DataFrame to CSV file
    df.to_csv(csv_path, index=False, header=True)
    print(f"Data has been saved to {csv_path}")



# def readTestDataFromPkl(pkl_path, num=0):
#     '''
#     从pkl格式的文件中读出指定num个作为milvus来源的数据resultDict
#     :pkl_path: 待处理的pkl地址
#     :num: 读出的数据条数
#     '''
#     pklDataList = readPklFile(pkl_path)
#     resultDict = {}
#     inputList = []
#     outputList = []
#     lengthList = []
#     idList = []
#     if(num <= 0 or num > len(pklDataList)):
#         num = len(pklDataList)
#     for i in range(num):
#         src = pklDataList[i].get("src")
#         tgt = pklDataList[i].get("tgt")
#         length = pklDataList[i].get("lengths")
#         id = pklDataList[i].get("id")
#         inputList.append(src)
#         outputList.append(tgt)
#         lengthList.append(length)
#         idList.append(id)
#     resultDict["inputList"] = inputList
#     resultDict["outputList"] = outputList
#     resultDict["lengthList"] = lengthList
#     resultDict["idList"] = idList
#     resultDict["num"] = num
#     resultDict["partDictOfSpecifyNum"] = pklDataList[:num]
#     return resultDict

def readSpecifyNumDataFromPkl(pkl_path, num=0):
    '''
    从pkl格式的文件中读出指定num个作为milvus来源的数据resultDict
    :pkl_path: 待处理的pkl地址
    :num: 读出的数据条数
    '''
    pklDataList = readPklFile(pkl_path)
    resultDict = {}
    inputList = []
    outputList = []
    lengthList = []
    idList = []
    if(num <= 0 or num > len(pklDataList)):
        num = len(pklDataList)
    for i in range(num):
        src = pklDataList[i].get("src")
        tgt = pklDataList[i].get("tgt")
        length = pklDataList[i].get("lengths")
        id = pklDataList[i].get("id")
        inputList.append(src)
        outputList.append(tgt)
        lengthList.append(length)
        idList.append(id)
    resultDict["inputList"] = inputList
    resultDict["outputList"] = outputList
    resultDict["lengthList"] = lengthList
    resultDict["idList"] = idList
    resultDict["num"] = num
    resultDict["partDictOfSpecifyNum"] = pklDataList[:num]
    return resultDict

def readDictDataFromPkl(pkl_path, num=0):
    pklDataList = readPklFile(pkl_path)
    resultDict = {}
    inputList = []
    outputList = []
    lengthList = []
    idList = []
    if(num <= 0 or num > len(pklDataList)):
        num = len(pklDataList)
    for i in range(num):
        src = pklDataList[i].get("src")
        tgt = pklDataList[i].get("tgt")
        length = pklDataList[i].get("lengths")
        id = pklDataList[i].get("id")
        if(length > 128):
            src = src[:127]
            tgt = tgt[:127]
            length = 128
        inputList.append(src)
        outputList.append(tgt)
        lengthList.append(length)
        idList.append(id)
    resultDict["inputList"] = inputList
    resultDict["outputList"] = outputList
    resultDict["lengthList"] = lengthList
    resultDict["idList"] = idList
    resultDict["num"] = num
    resultDict["partDictOfSpecifyNum"] = pklDataList[:num]
    return resultDict

def readVectorDatabaseDataFromPkl(pkl_path, num=0):
    '''
    从pkl格式的文件中读出指定num个作为milvus来源的数据resultDict
    :pkl_path: 待处理的pkl地址
    :num: 读出的数据条数
    '''
    pklDataList = readPklFile(pkl_path)
    resultDict = {}
    inputList = []
    outputList = []
    lengthList = []
    idList = []
    if(num <= 0 or num > len(pklDataList)):
        num = len(pklDataList)
    for i in range(num):
        src = pklDataList[i].get("src")
        tgt = pklDataList[i].get("tgt")
        length = pklDataList[i].get("lengths")
        id = pklDataList[i].get("id")
        if(length > 128):
            src = src[:127]
            tgt = tgt[:127]
            length = 128
        inputList.append(src)
        outputList.append(tgt)
        lengthList.append(length)
        idList.append(id)
    resultDict["inputList"] = inputList
    resultDict["outputList"] = outputList
    resultDict["lengthList"] = lengthList
    resultDict["idList"] = idList
    resultDict["num"] = num
    resultDict["partDictOfSpecifyNum"] = pklDataList[:num]
    return resultDict


def getExamplesRelatedToDataset(test_dataset_path,chat_shot_num,test_num):
    '''
    param:
        file_path: csv的路径
        chat_shot_num: 样本数目
    function:
        根据数据集和path选择对应的样本
    '''
    data_prefix = "dataset/finetuneDataset/"
    if "15" in test_dataset_path:
        finetune_data_name = "sighan15/sighan15_finetune_data.csv"
    elif "14" in test_dataset_path:
        finetune_data_name = "sighan14/sighan14_finetune_data.csv"
    elif "13" in test_dataset_path:
        finetune_data_name = "sighan13/sighan13_finetune_data.csv"
    elif "OAD" in test_dataset_path or "oad" in test_dataset_path:
        finetune_data_name = "oad_finetune_data.csv"
    elif "cscd" in test_dataset_path or "CSCD" in test_dataset_path:
        finetune_data_name = "cscd_ime_finetune_data.csv"
    file_path = data_prefix+finetune_data_name
    examples = readInputAndOutputSentenceFromCsv(file_path,chat_shot_num)

    new_list = [examples.copy() for _ in range(test_num)]
    return new_list

def getPredResult(messageResult):
    '''获取回答中的各个信息'''

    # 根据换行符分割成三行信息
    if (messageResult == None):
        return ""
    result_lines = messageResult.split("\n")

    # 解析每行信息
    try:
        for line in result_lines:
            if "错误汉字" in line:
                wrongChar = line.strip().split(":")[1]
            if "中文句子:" in line or "处理后的中文句子:" in line:
                predStr = line.strip().split(":")[1]
            if "字符串长度:" in line:
                preLength = line.strip().split(":")[1]
            if "相同" in line:
                equalLength = line.strip().split(":")[1]
        predSentence = predStr.strip()
    except Exception as e:
        print(e)
        predSentence = ""
    return predSentence

def readInputAndOutputSentenceFromCsv(file_path,chat_shot_num):
    '''
    param:
        file_path: csv的路径
        chat_shot_num: 样本数目
    function:
        读取csv中数据到dict的功能
    '''
    result = []

    try:
        # 打开并读取 CSV 文件
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            
            # 遍历行并存入结果列表，限制到指定数量
            for index, row in enumerate(reader):
                if index >= chat_shot_num:  # 达到样本数目限制时停止读取
                    break
                result.append({
                    "input": row["input"],   # 假设 CSV 文件的列名为 'input'
                    "output": row["output"]  # 假设 CSV 文件的列名为 'output'
                })
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")

    return result



def storageDataToCsvFile(dict_list,file_path):
    '''
    param:
        dict_list:元素列表，每个元素都是字典
        file_path:用于存放的csv的文件路径
    function:
        存储数据到csv
    '''
    # 检查字典列表是否为空
    if not dict_list:
        print("The provided dictionary list is empty. No CSV file was created.")
        return

    # 从第一个字典获取字段名
    field_names = dict_list[0].keys()
    try:
        # 打开目标文件并写入 CSV 数据
        with open(file_path, mode='w', encoding='utf-8', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            # 写入表头
            writer.writeheader()
            # 写入数据行
            writer.writerows(dict_list)
        print(f"Data successfully stored in CSV file: {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to CSV file: {e}")


    
