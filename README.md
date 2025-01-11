# RagID
This repository contains the source code for COLING2025 paper "Retrieval-Augmented Generation for Large Language Model based Few-shot Chinese Spell Checking".


## Download the file
The content of the 'RagID/dataset/' path is in the "https://1drv.ms/u/s!AoXIDmMNhCV5bHWzgt9DQKfo48c", after getting the project in github, please continue to download the file and extract it to the specified location

## Follow the steps below to initialize the project
```
$ conda create -n csc python=3.7.16
$ pip install -r requirements.txt 
$ cd milvus
$ docker compose up -d
$ cd ..
$ python preparation.py
$ python test.py
```

## If you need to do more testing, change the settings in the test.py
```
    few_shot_num = 4  # ！！！ 用于模型进行纠错学习的例子数目
    test_num = 100  # ！！！If you want to test all the data in the dataset, test_num change it to 0
    model_name = "gpt3.5"   #！！！Models used   gpt3.5   glm4
    stragety = "rag+ids"  #！！！Strategies used
    vecFunction = bgeLargeZhSentenceTrans   # Vectorization methods
    fieldName = "embedding"  # The vector database field used to make the query
    # For the vector database dataset used for this query, 58w represents the scale and 1 represents the first collections
    collectionNameList = ['CSCData58w_0', 'CSCData58w_1', 'CSCData58w_2', 'CSCData58w_3', 'CSCData58w_4', 'CSCData58w_5', 'CSCData58w_6', 'CSCData58w_7', 'CSCData58w_8', 'CSCData58w_9', 'CSCData58w_10', 'CSCData58w_11', 'CSCData58w_12', 'CSCData58w_13', 'CSCData58w_14', 'CSCData58w_15', 'CSCData58w_16', 'CSCData58w_17', 'CSCData58w_18', 'CSCData58w_19', 'CSCData58w_20', 'CSCData58w_21', 'CSCData58w_22', 'CSCData58w_23', 'CSCData58w_24', 'CSCData58w_25', 'CSCData58w_26', 'CSCData58w_27', 'CSCData58w_28', 'CSCData58w_29', 'CSCData58w_30', 'CSCData58w_31', 'CSCData58w_32', 'CSCData58w_33', 'CSCData58w_34', 'CSCData58w_35', 'CSCData58w_36', 'CSCData58w_37', 'CSCData58w_38', 'CSCData58w_39', 'CSCData58w_40', 'CSCData58w_41', 'CSCData58w_42', 'CSCData58w_43', 'CSCData58w_44', 'CSCData58w_45', 'CSCData58w_46', 'CSCData58w_47', 'CSCData58w_48', 'CSCData58w_49', 'CSCData58w_50']

    testDataset = "sighan15" # The dataset for this test.  sighan15,sighan14,sighan13,oad,cscd
```

## The "2024-02-15-preview" version of GPT3.5 was used for the experiment, and since the API of Azure OpenAI was not continued to be used, it is now changed to the official interface of OpenAI, and the model uses openai/gpt-3.5-turbo, which may have a small gap with the experimental results


