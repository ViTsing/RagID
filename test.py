# testGLM4
from tools.chatWithLLMTools import chatWithGLM4,chatWithGpt3Point5
from cscStrategy import ragAndIdsStrategyFunction,ragStrategyFunction,idsStrategyFunction,defaultStrategyFunction
from tools.vectorDatabaseTools import bgeLargeZhSentenceTrans


def testInDatasets(few_shot_num,test_num,stragety,model_name,vecFunction,fieldName,collectionNameList,testDataset):
    '''
    params:
        chat_shot_num: prompt中使用的样本数目
        stragety: csc使用的纠错策略，可选项有rag+ids,rag,ids,default
        model_name: 模型名称，可选项有glm4,gpt3.5
    function:
        功能：
    '''
    if testDataset == "sighan15":
        test_dataset_tsv_path = "dataset/testDataset/test_sighan15.tsv"
        test_dataset_path = "dataset/testDataset/test_sighan15.pkl"
    elif testDataset == "sighan14":
        test_dataset_tsv_path = "dataset/testDataset/test_sighan14.tsv"
        test_dataset_path = "dataset/testDataset/test_sighan14.pkl"
    elif testDataset == "sighan13":
        test_dataset_tsv_path = "dataset/testDataset/test_sighan13.tsv"
        test_dataset_path = "dataset/testDataset/test_sighan13.pkl"
    elif testDataset == "oad":
        test_dataset_tsv_path = "dataset/testDataset/OAD500.tsv"
        test_dataset_path = "dataset/testDataset/OAD500.pkl"
    elif testDataset == "cscd":
        test_dataset_tsv_path = "dataset/testDataset/CSCD_IME_500Test.tsv"
        test_dataset_path = "dataset/testDataset/CSCD_IME_500Test.pkl"
    else:
        test_dataset_tsv_path = "dataset/testDataset/test_sighan15.tsv"
        test_dataset_path = "dataset/testDataset/test_sighan15.pkl"


    if model_name == "gpt3.5":
        chatAI = chatWithGpt3Point5
    else:
        chatAI = chatWithGLM4

    chatStrategy = ragAndIdsStrategyFunction
    use_rag = False
    use_ids = False
    if stragety == "rag+ids":
        use_rag = True
        use_ids = True
    elif stragety == "rag":
        use_rag = True
        use_ids = False
    elif stragety == "ids":
        use_rag = False
        use_ids = True
    chatStrategy(chatAI,few_shot_num,test_num,test_dataset_path,test_dataset_tsv_path,use_rag,use_ids,vecFunction,fieldName,collectionNameList)


if __name__ == "__main__":
    few_shot_num = 4
    test_num = 100  # ！！！If you want to test all the data in the dataset, test_num change it to 0
    model_name = "gpt3.5"
    stragety = "rag+ids"
    vecFunction = bgeLargeZhSentenceTrans
    fieldName = "embedding"
    collectionNameList = ['CSCData58w_0', 'CSCData58w_1', 'CSCData58w_2', 'CSCData58w_3', 'CSCData58w_4', 'CSCData58w_5', 'CSCData58w_6', 'CSCData58w_7', 'CSCData58w_8', 'CSCData58w_9', 'CSCData58w_10', 'CSCData58w_11', 'CSCData58w_12', 'CSCData58w_13', 'CSCData58w_14', 'CSCData58w_15', 'CSCData58w_16', 'CSCData58w_17', 'CSCData58w_18', 'CSCData58w_19', 'CSCData58w_20', 'CSCData58w_21', 'CSCData58w_22', 'CSCData58w_23', 'CSCData58w_24', 'CSCData58w_25', 'CSCData58w_26', 'CSCData58w_27', 'CSCData58w_28', 'CSCData58w_29', 'CSCData58w_30', 'CSCData58w_31', 'CSCData58w_32', 'CSCData58w_33', 'CSCData58w_34', 'CSCData58w_35', 'CSCData58w_36', 'CSCData58w_37', 'CSCData58w_38', 'CSCData58w_39', 'CSCData58w_40', 'CSCData58w_41', 'CSCData58w_42', 'CSCData58w_43', 'CSCData58w_44', 'CSCData58w_45', 'CSCData58w_46', 'CSCData58w_47', 'CSCData58w_48', 'CSCData58w_49', 'CSCData58w_50']
    testDataset = "sighan15"
    testInDatasets(few_shot_num,test_num,stragety,model_name,vecFunction,fieldName,collectionNameList,testDataset)