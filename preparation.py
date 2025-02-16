# 开始前的准备
from tools.vectorDatabaseTools import collectionForBigDataset,bgeLargeZhSentenceTrans
# 数据路径
vector_data_path = "dataset/vectorDatabaseDataset/mergeData58w.pkl"
# 创建1w,2w,5w,10w,20w,30w,40w,50w,58w的向量数据库集合
vecFunction = bgeLargeZhSentenceTrans
name_prefix_list = ["CSCData1w","CSCData2w","CSCData5w","CSCData10w","CSCData20w","CSCData30w","CSCData40w","CSCData50w","CSCData58w"]
number_list = [10000,20000,50000,100000,200000,300000,400000,500000,580000]
embeddingLength = 1024
with open('vectorCollectionList.txt', 'w') as file:
    for (seriesName,number) in zip(name_prefix_list,number_list):
        collection_list = collectionForBigDataset(seriesName,vecFunction,vector_data_path,number,embeddingLength)
        file.write(str(collection_list) + '\n')
file.close()
# 得到向量数据库的集合名称
# 1w条 ['CSCData1w_0']
# 2w条 ['CSCData2w_0', 'CSCData2w_1']
# 5w条 ['CSCData5w_0', 'CSCData5w_1', 'CSCData5w_2', 'CSCData5w_3', 'CSCData5w_4']
# 10w条 ['CSCData10w_0', 'CSCData10w_1', 'CSCData10w_2', 'CSCData10w_3', 'CSCData10w_4', 'CSCData10w_5', 'CSCData10w_6', 'CSCData10w_7', 'CSCData10w_8', 'CSCData10w_9']
# 20w条 ['CSCData20w_0', 'CSCData20w_1', 'CSCData20w_2', 'CSCData20w_3', 'CSCData20w_4', 'CSCData20w_5', 'CSCData20w_6', 'CSCData20w_7', 'CSCData20w_8', 'CSCData20w_9', 'CSCData20w_10', 'CSCData20w_11', 'CSCData20w_12', 'CSCData20w_13', 'CSCData20w_14', 'CSCData20w_15', 'CSCData20w_16', 'CSCData20w_17', 'CSCData20w_18', 'CSCData20w_19']
# 30w条 ['CSCData30w_0', 'CSCData30w_1', 'CSCData30w_2', 'CSCData30w_3', 'CSCData30w_4', 'CSCData30w_5', 'CSCData30w_6', 'CSCData30w_7', 'CSCData30w_8', 'CSCData30w_9', 'CSCData30w_10', 'CSCData30w_11', 'CSCData30w_12', 'CSCData30w_13', 'CSCData30w_14', 'CSCData30w_15', 'CSCData30w_16', 'CSCData30w_17', 'CSCData30w_18', 'CSCData30w_19', 'CSCData30w_20', 'CSCData30w_21', 'CSCData30w_22', 'CSCData30w_23', 'CSCData30w_24', 'CSCData30w_25', 'CSCData30w_26', 'CSCData30w_27', 'CSCData30w_28', 'CSCData30w_29']
# 40w条 ['CSCData40w_0', 'CSCData40w_1', 'CSCData40w_2', 'CSCData40w_3', 'CSCData40w_4', 'CSCData40w_5', 'CSCData40w_6', 'CSCData40w_7', 'CSCData40w_8', 'CSCData40w_9', 'CSCData40w_10', 'CSCData40w_11', 'CSCData40w_12', 'CSCData40w_13', 'CSCData40w_14', 'CSCData40w_15', 'CSCData40w_16', 'CSCData40w_17', 'CSCData40w_18', 'CSCData40w_19', 'CSCData40w_20', 'CSCData40w_21', 'CSCData40w_22', 'CSCData40w_23', 'CSCData40w_24', 'CSCData40w_25', 'CSCData40w_26', 'CSCData40w_27', 'CSCData40w_28', 'CSCData40w_29', 'CSCData40w_30', 'CSCData40w_31', 'CSCData40w_32', 'CSCData40w_33', 'CSCData40w_34', 'CSCData40w_35', 'CSCData40w_36', 'CSCData40w_37', 'CSCData40w_38', 'CSCData40w_39']
# 50w条 ['CSCData50w_0', 'CSCData50w_1', 'CSCData50w_2', 'CSCData50w_3', 'CSCData50w_4', 'CSCData50w_5', 'CSCData50w_6', 'CSCData50w_7', 'CSCData50w_8', 'CSCData50w_9', 'CSCData50w_10', 'CSCData50w_11', 'CSCData50w_12', 'CSCData50w_13', 'CSCData50w_14', 'CSCData50w_15', 'CSCData50w_16', 'CSCData50w_17', 'CSCData50w_18', 'CSCData50w_19', 'CSCData50w_20', 'CSCData50w_21', 'CSCData50w_22', 'CSCData50w_23', 'CSCData50w_24', 'CSCData50w_25', 'CSCData50w_26', 'CSCData50w_27', 'CSCData50w_28', 'CSCData50w_29', 'CSCData50w_30', 'CSCData50w_31', 'CSCData50w_32', 'CSCData50w_33', 'CSCData50w_34', 'CSCData50w_35', 'CSCData50w_36', 'CSCData50w_37', 'CSCData50w_38', 'CSCData50w_39', 'CSCData50w_40', 'CSCData50w_41', 'CSCData50w_42', 'CSCData50w_43', 'CSCData50w_44', 'CSCData50w_45', 'CSCData50w_46', 'CSCData50w_47', 'CSCData50w_48', 'CSCData50w_49']
# 58w条 ['CSCData58w_0', 'CSCData58w_1', 'CSCData58w_2', 'CSCData58w_3', 'CSCData58w_4', 'CSCData58w_5', 'CSCData58w_6', 'CSCData58w_7', 'CSCData58w_8', 'CSCData58w_9', 'CSCData58w_10', 'CSCData58w_11', 'CSCData58w_12', 'CSCData58w_13', 'CSCData58w_14', 'CSCData58w_15', 'CSCData58w_16', 'CSCData58w_17', 'CSCData58w_18', 'CSCData58w_19', 'CSCData58w_20', 'CSCData58w_21', 'CSCData58w_22', 'CSCData58w_23', 'CSCData58w_24', 'CSCData58w_25', 'CSCData58w_26', 'CSCData58w_27', 'CSCData58w_28', 'CSCData58w_29', 'CSCData58w_30', 'CSCData58w_31', 'CSCData58w_32', 'CSCData58w_33', 'CSCData58w_34', 'CSCData58w_35', 'CSCData58w_36', 'CSCData58w_37', 'CSCData58w_38', 'CSCData58w_39', 'CSCData58w_40', 'CSCData58w_41', 'CSCData58w_42', 'CSCData58w_43', 'CSCData58w_44', 'CSCData58w_45', 'CSCData58w_46', 'CSCData58w_47', 'CSCData58w_48', 'CSCData58w_49', 'CSCData58w_50']

