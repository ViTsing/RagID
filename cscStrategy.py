# strategy Choice
import os
from tools.metricCscTools import metricThroughtTsvFile
from tools.fileOperationTools import readSpecifyNumDataFromPkl,getExamplesRelatedToDataset,getFormatFewShotCommentResult,getPredResult
from tools.vectorDatabaseTools import getExamplesRelatedToDatasetFromVectorDatabase
def ragAndIdsStrategyFunction(chatAI,few_shot_num,test_num,test_dataset_path,test_dataset_tsv_path,use_rag,use_ids,vecFunction,fieldName,collectionNameList):
    # 从数据集中拿出测试用句，测试用句id，测试用句答案
    test_dataset_dict = readSpecifyNumDataFromPkl(test_dataset_path,test_num)
    test_input_sentence_list = test_dataset_dict.get("inputList")
    test_input_id_list = test_dataset_dict.get("idList")

    file_name_prefix  = os.path.splitext(os.path.basename(test_dataset_path))[0]
    saveFileExampelsName = f"{file_name_prefix}examples.log"
    saveFileProcessName = f"{file_name_prefix}process.log"
    os.remove(saveFileExampelsName) if os.path.exists(saveFileExampelsName) else None
    os.remove(saveFileProcessName) if os.path.exists(saveFileProcessName) else None

    # 判断是使用固定的样例还是不固定的样例
    if use_rag: 
        print("start_query")
        examples = getExamplesRelatedToDatasetFromVectorDatabase(test_input_sentence_list,few_shot_num,vecFunction,fieldName,collectionNameList)
        print("finish_query")
    else:
        examples = getExamplesRelatedToDataset(test_dataset_path,few_shot_num,test_num)

        
    # 按照策略测试
    test_output_list = []
    for i in range(len(test_input_id_list)):
        max_iteration = 5
        iteration_time = 0
        comment = ""
        discriminator_pass = False
        id = test_input_id_list[i]
        input = test_input_sentence_list[i]
        now_example_list = examples[i]
        # 对单个句子进行纠错
        middle_input = input
        middle_output = ""
        while(iteration_time < max_iteration and discriminator_pass==False):
            # RAG
            
            chat_message = getPromptFromExamples(input,middle_output,now_example_list,comment,use_rag,use_ids)
            csc_result = chatAI(chat_message)
            middle_output = getPredResult(csc_result)
            # print(chat_message)
            # IDS
            if use_ids:
                discriminator_message = getCommentFromDiscriminator(middle_input,middle_output)
                # print(discriminator_message)
                comment = chatAI(discriminator_message)
                discriminator_pass,comment =  getFormatFewShotCommentResult(comment,middle_input,middle_output)
                print(middle_input)
                print(middle_output)
                print(comment)
                print(discriminator_pass)
            else:
                iteration_time = max_iteration
                discriminator_pass = True
            iteration_time += 1
            # 保存纠错过程
            saveCSCProgress(id,input,middle_output,comment,discriminator_pass,iteration_time,saveFileProcessName)
        output = middle_output
        if discriminator_pass == False:
            output = input
        saveCSCExamples(id,now_example_list,saveFileExampelsName)
        test_output_list.append(output)
    # 评估纠错结果，需要csv文件
    score = metricThroughtTsvFile(test_input_id_list, test_input_sentence_list, test_output_list, test_dataset_tsv_path,test_num)
    print(score)


def saveCSCExamples(id, examples, file_name):
    '''
    params:
        id: 句子id
        examples: 句子相关的信息
        file_name: 文件名
    function:
        存储信息到file_name代表的文件中，存储的方式是：id和examples单独存放在两行并空一行。
    '''
    
    # 打开文件，以追加模式写入（如果文件不存在则创建）
    with open(file_name, "a", encoding="utf-8") as file:
        # 写入句子id
        file.write(f"id:{id}\n")
        # 写入examples
        file.write(f"examples:{examples}\n")
        # 空一行作为分隔
        file.write("\n")

def saveCSCProgress(id, input, output, comment, discriminator_pass, iteration_time, file_name):
    '''
    params:
        id: 句子的id,
        input: 句子的内容,
        output: 修改后句子的内容,
        comment: 此次修改的评论,
        discriminator_pass: 是否通过修改成功判定,
        iteration_time: 修改迭代次数,
        file_name: 用于存储的文件名
    function:
        存储信息到.log文件中，存储的方式是：id, input, output, comment, discriminator_pass, iteration_time 各自存一行，并加一行空行
    '''
    # 打开文件，以追加模式写入（如果文件不存在则创建）
    with open(file_name, "a", encoding="utf-8") as file:
        # 写入每个字段，每个字段单独一行
        file.write(f"id: {id}\n")
        file.write(f"input: {input}\n")
        file.write(f"output: {output}\n")
        file.write(f"comment: {comment}\n")
        file.write(f"discriminator_pass: {discriminator_pass}\n")
        file.write(f"iteration_time: {iteration_time}\n")
        # 空一行作为分隔
        file.write("\n")




def ragStrategyFunction(chatAI,chatStrategy,chat_shot_num,test_dataset_path):
    ""

def idsStrategyFunction(chatAI,chatStrategy,chat_shot_num,test_dataset_path):
    ""

def defaultStrategyFunction(chatAI,chatStrategy,chat_shot_num,test_dataset_path):
    ""

def getPromptFromExamples(current_query_sentence,middle_output,examples,comment,use_rag,use_ids):
    '''

    '''

    # 角色定义
    define_character = "我想让你担任一个强大的中文拼写纠错模型，你的职责是对中文句子中的汉字进行拼写检查和纠错。"
    query_prompt = """
    待处理中文句子:{}
    待处理中文句子的字符串长度:{}
    处理指令:结合中文语言学规则、上下文信息对上文中的待处理中文句子进行检查，判断句子中是否错误使用了拼音相似或者形状相似的汉字，导致中文句子原本的语义发生改变。如果句子中有这种错误使用的汉字，就找到正确的汉字，对错误汉字进行一对一替换。有几个错误汉字就替换几次，替换操作发生前后中文句子的字符串长度不发生改变；如果没有错误使用的汉字，则不进行替换。不对人名，地名等专有名词进行替换,如果上次纠错的结果不合理，此次纠错应该重新检查是否有错误或者纠错其他位置的字符。 
    上次纠错的结果:{}
    上次纠错的结果评价:{}

    结果模板:
    是否存在错误汉字:<>
    处理后的中文句子:<>
    处理后的中文句子的字符串长度:<>
    """
    answe_prompt = """
    是否存在错误汉字:{}
    处理后的中文句子:{}
    处理后的中文句子的字符串长度:{}
    """
    extra_example = {"input":"他们准备了很多吃的东西","output":"他们准备了很多吃的东西"}
    if use_rag:
        examples.pop()
        examples.append(extra_example)
    # 消息列表定义
    messageText = []
    defineDict = {"role":"system","content":define_character}
    messageText.append(defineDict)
    # 历史对话伪造
    for exmaple in examples:
        comment = ""
        query_content = exmaple.get("input")
        answer_content = exmaple.get("output")
        if use_ids:
            if query_content == answer_content:
                comment = ""
            else:
                comment = ""
        QueryDict = {"role":"user","content":query_prompt.format(query_content,len(query_content),middle_output,comment)}
        messageText.append(QueryDict)
        answerJudge = "是" if query_content != answer_content else "否"
        AnswerDict = {"role":"assistant","content":answe_prompt.format(answerJudge,answer_content,len(answer_content))}
        messageText.append(AnswerDict)
    # 当前对话置入
    queryDictNow = {"role":"user","content":query_prompt.format(current_query_sentence,len(current_query_sentence),"","")}
    messageText.append(queryDictNow)
    return messageText



def getCommentFromDiscriminator(query,predict):
    define_prompt = "我想让你担任一个中文拼写纠错的评价模型，你的职责是结合相关领域知识，判断中文拼写纠错模型的纠错结果是否符合要求。"
    query_prompt = """
    纠错前的中文句子:{}
    纠错前的中文句子长度:{}

    纠错后的中文句子:{}
    纠错后的中文句子长度:{}

    评价标准:结合中文拼写纠错的知识，判断当前发生的纠错是否合理，纠错后的中文句子中是否仍然存在错误？纠错后的句子是否更加符合语义？纠错前后句子长度是否一致？只输出结果模板的内容。

    评价结果模板:
    纠错后的句子已经不存在字符错误:<>
    纠错后的句子符合语法逻辑:<>
    纠错后的句子和纠错前句子长度相等:<>
    """
    answer_prompt = """
    纠错后的句子已经不存在字符错误:{}
    纠错后的句子符合语法逻辑:{}
    纠错后的句子和纠错前句子长度相等:{}
    """
    query_examples =  [
        {"src":"我请你吃饭，可以吗？","tgt":"我请你吃饭，可以吗？"},
        {"src":"在公车上有很多人，所以我们没有位子可以座。","tgt":"在公车上有很多人，所以我们没有位子可以坐。"},
        {"src":"在补习班他昨天晚上到夜里两点还在读书，所以他一回家就累得睡着了。","tgt":"在补习班他昨天晚上到夜里两点还在读书，所以他一回家就累得不动了。"},
        {"src":"国务员办公厅关于引发深化医药卫生体制改个","tgt":"国务院办公厅关于引发深化医药卫生体制改革的。"}
    ]
    answer_examples = [
        {"dontHaveWrongChar":"是","moreReasonable":"是","equalLength":"是"},
        {"dontHaveWrongChar":"是","moreReasonable":"是","equalLength":"是"},
        {"dontHaveWrongChar":"否","moreReasonable":"否","equalLength":"是"},
        {"dontHaveWrongChar":"否","moreReasonable":"否","equalLength":"否"}
    ]
   # 定义内容
    # 找辅助样例4个
    messageText = []
    defineDict = {"role":"system","content":define_prompt}
    messageText.append(defineDict)
    # 历史对话
    for (query,answer) in zip(query_examples,answer_examples):
        query_content_src = query.get("src")
        query_content_tgt = query.get("tgt")
        QueryDict = {"role":"user","content":query_prompt.format(query_content_src,len(query_content_src),query_content_tgt,len(query_content_tgt))}
        messageText.append(QueryDict)

        dontHaveWrongChar = answer.get("dontHaveWrongChar")
        moreReasonable = answer.get("moreReasonable")
        equalLength = answer.get("equalLength")
        AnswerDict = {"role":"assistant","content":answer_prompt.format(dontHaveWrongChar,moreReasonable,equalLength)}
        messageText.append(AnswerDict)

    queryDictNow = {"role":"user","content":query_prompt.format(query,len(query),predict,len(predict))}
    messageText.append(queryDictNow)
    return messageText

# def createGPTMessageHistory(role_list,conversation_list):
#     messageText = []
#     for (role,conversation) in (role_list,conversation_list):
#         temp_dict = {"role":role,"content":conversation}
#         messageText.append(temp_dict)

# def createGLM4MessageHistory(role_list,conversation_list):
#     messageText = []
#     for (role,conversation) in (role_list,conversation_list):
#         if role == "assistant": role = "system"
#         temp_dict = {"role":role,"content":conversation}
#         messageText.append(temp_dict)