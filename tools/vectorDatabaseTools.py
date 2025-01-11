# 创建数据库
import numpy as np
from .fileOperationTools import readVectorDatabaseDataFromPkl,readDictDataFromPkl
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    db
)
import uuid

def collectionForBigDataset(seriesName,vecFunction,srcPath,number,embeddingLength):
    '''
    在milvus中创建多个collection
    :collectionName: 集合名称
    :vecFunction: 使用的向量化方法
    :srcPath: 数据来源
    :number: 当前集合中的数据条数
    :embeddingLength: 集合中向量长度
    '''
    # 从pkl中读取数据
    pklDict = readVectorDatabaseDataFromPkl(srcPath,number)
    outputList = pklDict.get("outputList")
    inputList = pklDict.get("inputList")
    lengthList = pklDict.get("lengthList")
    partDictOfSpecifyNum =  pklDict.get("partDictOfSpecifyNum")
    number =  pklDict.get("num")
    
    
    # 看看当前需要使用多少个Collection
    COLLECTION_SIZE = 10000
    collectionNum = number // COLLECTION_SIZE
    if number % COLLECTION_SIZE != 0:
        collectionNum += 1
        
    nameList = []
    for i in range(collectionNum):
        nameNow = seriesName + "_" + str(i)
        if(i+1 != collectionNum):
            collectionSizeNow = COLLECTION_SIZE
            inputData = inputList[i*COLLECTION_SIZE:(i+1)*COLLECTION_SIZE]
            outputData = outputList[i*COLLECTION_SIZE:(i+1)*COLLECTION_SIZE]
            numberDataDict = partDictOfSpecifyNum[i*COLLECTION_SIZE:(i+1)*COLLECTION_SIZE]
            lengthData = lengthList[i*COLLECTION_SIZE:(i+1)*COLLECTION_SIZE]
            idData = getUUIDList(collectionSizeNow)
            vectorList = vecFunction(outputData)
            
        else:
            collectionSizeNow = number - i * COLLECTION_SIZE
            inputData = inputList[i*COLLECTION_SIZE: number]
            outputData = outputList[i*COLLECTION_SIZE: number]
            numberDataDict = partDictOfSpecifyNum[i*COLLECTION_SIZE: number]
            lengthData = lengthList[i*COLLECTION_SIZE: number]
            idData = getUUIDList(collectionSizeNow)
            vectorList = vecFunction(outputData)
        if not isinstance(vectorList, np.ndarray):
            vectorList = vectorList.numpy()
        collectedName = createSingleCollection(nameNow,vecFunction,srcPath,collectionSizeNow,embeddingLength,idData,inputData,outputData,lengthData,vectorList)
        nameList.append(collectedName)
    print("已经创建的collection是",nameList)
    print("当前milvus中的collection有：",utility.list_collections())
    return nameList


def getUUIDList(num):
    '''
    得到指定个数的UUID
    '''
    idList = []
    for i in range(num):
        idList.append(str(uuid.uuid4()))
    return idList


def createSingleCollection(collectionName,vecFunction,srcPath,number,embeddingLength,idList,inputList,outputList,lengthList,vectorList):
    '''
    在milvus中创建一个collection
    :collectionName: 集合名称
    :vecFunction: 使用的向量化方法
    :srcPath: 数据来源
    :number: 当前集合中的数据条数
    :embeddingLength: 集合中向量长度
    '''
    # 连接服务
    con = connections.connect("default", host="localhost", port="19530")
    ls = db.list_database()
    print(ls)
    # 使用某个数据库
    db.using_database("default")
    utility.drop_collection(collectionName)
    # 删除后展示当前的数据库中有多少的collection
    print("创建前milvus中的collection有：",utility.list_collections())
    
    # 创建一个collection
    id = FieldSchema(
    name="id",
    dtype=DataType.VARCHAR,
    is_primary=True,
    max_length=200
    )
    input = FieldSchema(
    name="input",
    dtype=DataType.VARCHAR,
    max_length=128*3,
    default_value="Unknown"
    )
    output  = FieldSchema(
    name="output",
    dtype=DataType.VARCHAR,
    max_length=128*3,
    default_value="Unknown"
    )
    embedding = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim = embeddingLength
    )
    length = FieldSchema(
    name="length",
    dtype=DataType.INT64,
    default_value = 0
    )
    collection_schema = CollectionSchema(
    fields=[id, input, output, embedding, length],
    description="law info",
    enable_dynamic_field=True
    )

    collection = Collection(name=collectionName, schema=collection_schema,shards_num=1)

    data = [
        idList,
        inputList,
        outputList,
        vectorList,
        lengthList
    ]

    collection.insert(data)
    params = {
        "M": 16,                # 每个节点保持的最大连接数量
        "efConstruction": 200,  # 在构建时每个节点要保留的近邻节点数量
        "efSearch": 100          # 在搜索时每个节点要保留的近邻节点数量
    }

    # 设置索引
    index_params = {
        "metric_type":"IP", # 点积更加符合文本相似性COSINE,IP,L2
        "index_type":"HNSW",
        "params":params
    }
    collection.create_index(
        field_name="embedding", 
        index_params=index_params
    )
    print(f"{collectionName}创建成功, 该collection的来源是{srcPath},使用的向量化方法是{vecFunction},该collection的长度是{number}")
    return collectionName
    
    
def searchFromCollectionList(querySentenceList,querySentenceListDict,collectionNameList,topK,fieldName,vecFunction):
    '''
    get nearest topk sentences from specify collection
    :querySentenceList: 待查询的句子列表
    :collectionName: 集合名称
    :topK: 最近邻搜索的数目
    :fieldName: 向量所在的字段名称
    :vecFunction: 向量化方法
    :querySentenceListDict: 待查询的句子列表（dict形式）
    ''' 
    finalList = []
    querySentenceVectorList = vecFunction(querySentenceList)
    if not isinstance(querySentenceVectorList, np.ndarray):
        querySentenceVectorList = querySentenceVectorList.numpy()
    for collectionName in collectionNameList:
        resultList = serachFromOneCollection(querySentenceVectorList,collectionName,topK,fieldName)
        finalList.append(resultList)
    
    # 得到n个resultList，循环遍历len(resultList)比较
    allExampleList = []
    queryNum = len(querySentenceList)
    for i in range(queryNum):
        nowList = []
        for list in finalList:
            nowList.extend(list[i])
        sortedList = sorted(nowList, key=lambda x: x['distance'], reverse=True)
        topKSortedList = sortedList[:topK]
        allExampleList.append(topKSortedList)
    return allExampleList      


def serachFromOneCollection(querySentenceVectorList,collectionName,topK,fieldName):
    con = connections.connect("default", host="localhost", port="19530")
    ls = db.list_database()
    db.using_database("default")
    collection = Collection(collectionName)
    collection.load()
    search_params = {
        "metric_type": "IP", 
        "offset": 0, 
        "ignore_growing": False, 
        "nprobe": 10
    }
    # 进行查询
    results = collection.search(
        data=querySentenceVectorList,  # 向量
        anns_field=fieldName,  # 查询的字段名
        # the sum of `offset` in `param` and `limit` 
        # should be less than 16384.
        param=search_params,
        limit=topK,
        # expr=None, # 查询表达式，暂时不知道是什么
        # set the names of the fields you want to 
        # retrieve from the search result.
        output_fields=['id','input','output','length'],
        consistency_level="Strong"
    )
    # 搜索完成后，需要释放Milvus中加载的集合以减少内存消耗
    collection.release()
    result_list = []
    for result in results:
        temp_list = []
        for x in result:
            result_dict = {}
            result_dict['input'] = x.entity.get('input')
            result_dict['output'] = x.entity.get('output')
            result_dict['id'] = x.entity.get('id')
            result_dict['distance'] = x.distance
            temp_list.append(result_dict)
        temp_list = sorted(temp_list, key=lambda x: x['distance'], reverse=True)
        result_list.append(temp_list)
    return result_list
    
def deleteAllCollection():
    '''
    删除milvus中的所有数据集合
    '''
    con = connections.connect("default", host="localhost", port="19530")
    ls = db.list_database()
    print(1)
    print(ls)
    # 使用某个数据库
    db.using_database("default")

    # 删除前
    print(utility.list_collections())
    collectionList = utility.list_collections()
    # utility.drop_collection("law_collection_fastText_200")
    for collectionName in collectionList:
        utility.drop_collection(collectionName)
    # 删除后
    print(utility.list_collections())
    
    
def getAllCollection():
    '''
    删除milvus中的所有数据集合
    '''
    con = connections.connect("default", host="localhost", port="19530")
    ls = db.list_database()
    print(ls)
    # 使用某个数据库
    db.using_database("default")

    # 删除前
    print(utility.list_collections())

# 向量化方法





def getExamplesRelatedToDatasetFromVectorDatabase(test_input_sentence_list,chat_shot_num,vecFunction,fieldName,collectionNameList):
    '''
    get nearest topk sentences from specify collection
    :querySentenceList: 待查询的句子列表
    :collectionName: 集合名称
    :topK: 最近邻搜索的数目
    :fieldName: 向量所在的字段名称
    :vecFunction: 向量化方法
    :querySentenceListDict: 待查询的句子列表（dict形式）
    ''' 
    topK = chat_shot_num
    finalList = []
    querySentenceVectorList = vecFunction(test_input_sentence_list)
    print("start_vec")
    if not isinstance(querySentenceVectorList, np.ndarray):
        querySentenceVectorList = querySentenceVectorList.numpy()
    print("finding")
    for collectionName in collectionNameList:
        print("...")
        resultList = serachFromOneCollection(querySentenceVectorList,collectionName,chat_shot_num,fieldName)
        finalList.append(resultList)
    
    # 得到n个resultList，循环遍历len(resultList)比较
    allExampleList = []
    queryNum = len(test_input_sentence_list)
    for i in range(queryNum):
        nowList = []
        for list in finalList:
            nowList.extend(list[i])
        sortedList = sorted(nowList, key=lambda x: x['distance'], reverse=True)
        topKSortedList = sortedList[:topK]
        allExampleList.append(topKSortedList)
    return allExampleList  


def bgeLargeZhSentenceTrans(sentences):
    '''
    models with sentence-transformers:
    bge-large-zh模型向量化方法  1024维
    :sentences: 待向量化的句子列表
    # :dictTypeOfSentences: dict格式的待向量化的句子列表
    '''
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    embeddings_1 = model.encode(sentences, normalize_embeddings=True)
    return embeddings_1

def collectionForBigDataset(seriesName,vecFunction,srcPath,number,embeddingLength):
    '''
    在milvus中创建多个collection
    :collectionName: 集合名称
    :vecFunction: 使用的向量化方法
    :srcPath: 数据来源
    :number: 当前集合中的数据条数
    :embeddingLength: 集合中向量长度
    '''
    # 从pkl中读取数据
    pklDict = readDictDataFromPkl(srcPath,number)
    outputList = pklDict.get("outputList")
    inputList = pklDict.get("inputList")
    lengthList = pklDict.get("lengthList")
    partDictOfSpecifyNum =  pklDict.get("partDictOfSpecifyNum")
    number =  pklDict.get("num")
    
    
    # 看看当前需要使用多少个Collection
    COLLECTION_SIZE = 10000
    collectionNum = number // COLLECTION_SIZE
    if number % COLLECTION_SIZE != 0:
        collectionNum += 1
        
    nameList = []
    for i in range(collectionNum):
        nameNow = seriesName + "_" + str(i)
        if(i+1 != collectionNum):
            collectionSizeNow = COLLECTION_SIZE
            inputData = inputList[i*COLLECTION_SIZE:(i+1)*COLLECTION_SIZE]
            outputData = outputList[i*COLLECTION_SIZE:(i+1)*COLLECTION_SIZE]
            numberDataDict = partDictOfSpecifyNum[i*COLLECTION_SIZE:(i+1)*COLLECTION_SIZE]
            lengthData = lengthList[i*COLLECTION_SIZE:(i+1)*COLLECTION_SIZE]
            idData = getUUIDList(collectionSizeNow)
            vectorList = vecFunction(outputData)
            
        else:
            collectionSizeNow = number - i * COLLECTION_SIZE
            inputData = inputList[i*COLLECTION_SIZE: number]
            outputData = outputList[i*COLLECTION_SIZE: number]
            numberDataDict = partDictOfSpecifyNum[i*COLLECTION_SIZE: number]
            lengthData = lengthList[i*COLLECTION_SIZE: number]
            idData = getUUIDList(collectionSizeNow)
            vectorList = vecFunction(outputData)
        if not isinstance(vectorList, np.ndarray):
            vectorList = vectorList.numpy()
        collectedName = createSingleCollection(nameNow,vecFunction,srcPath,collectionSizeNow,embeddingLength,idData,inputData,outputData,lengthData,vectorList)
        nameList.append(collectedName)
    print("已经创建的collection是",nameList)
    print("当前milvus中的collection有：",utility.list_collections())
    return nameList
if __name__ == "__main__":
        getAllCollection()