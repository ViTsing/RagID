# Obtain surface features of the data set
import csv
import os
from tools.fileOperationTools import getFilesFromFolder,save_to_csv, readCSVFile

def getSurfaceFeature(file_folder,output_path):
    '''
    param:
        file_folder: folder where csv files are stored
        output_path: path of result file
    function:
        Get the surface data features of the dataset and output them to a file
    '''
    header_list = [
        'Dataset Name', 
        'Sentence Number', 
        'Correct Sentence Number', 
        'Incorrect Sentence Number', 
        'Max Length of Sentence', 
        'Min Length of Sentence', 
        'Average Length of Sentence', 
        'Total Number of Wrong Character', 
        'Average Number of Wrong Character per sentence', 
        'Average Number of Characters between two Wrong character', 
        'Sentence Number containing wrong characters 他她它', 
        'Sentence Number containing wrong characters 的得地', 
        'Character Num about wrong characters 他她它', 
        'Character Num about wrong characters 的得地'
    ]
    # Initialize a result list where each element is a list of a specific column's data
    content_list = [
        [],  # Dataset Name
        [],  # Sentence Number
        [],  # Correct Sentence Number
        [],  # Incorrect Sentence Number
        [],  # Max Length of Sentence
        [],  # Min Length of Sentence
        [],  # Average Length of Sentence
        [],  # Total Number of Wrong Character
        [],  # Average Number of Wrong Character per sentence
        [],  # Average Number of Characters between two Wrong character
        [],  # Sentence Number containing wrong characters 他她它
        [],  # Sentence Number containing wrong characters 的得地
        [],  # Character Num about wrong characters 他她它
        [],  # Character Num about wrong characters 的得地
    ]
    # sentenceNumList
    file_list = getFilesFromFolder(file_folder)
    file_list = [file for file in file_list if file.endswith('.csv')]
    for file in file_list: 
        header,content = readCSVFile(file)
        sentenceNum = 0  # total number of sentences
        rightSentenceNum = 0 # total number of sentences without errors
        wrongSentenceNum = 0 # total number of sentences containing errors
        maxSentencelength = 0 # maximum sentence length
        minSentencelength = 1000 # minimum sentence length
        averageSentenceLength = 0 # average sentence length
        allWrongCharNum = 0 # total character errors
        totalSentencesLength = 0  # total sentence length
        deWrongSentenceNum = 0 # number of sentences containing the error '的' '得' '地'
        deWrongCharNum = 0  # number of wrong characters related to '的' '得' '地'
        taWrongSentenceNum = 0   # number of sentences containing the error '他' '她' '它'
        taWrongCharNum = 0  # number of wrong characters related to '他' '她' '它'
        for row in content:
            id, label, src, tgt = row
            allWrongCharNum += int(label)
            totalSentencesLength  += len(src)
            sentenceNum += 1
            if(src == tgt):
                rightSentenceNum += 1
            else:
                wrongSentenceNum += 1
                isDE,nowSentenceDeNum = judgeWrongType(src,tgt,['的','得','地'])
                if isDE:
                    deWrongSentenceNum += 1
                    deWrongCharNum += nowSentenceDeNum
                isTa,nowSentenceTaNum = judgeWrongType(src,tgt,['他','她','它'])
                if isTa:
                    taWrongSentenceNum += 1
                    taWrongCharNum += nowSentenceTaNum
            if maxSentencelength < len(src):
                maxSentencelength = len(src)
            if minSentencelength > len(src):
                minSentencelength = len(src)
            
        averageSentenceLength = round(totalSentencesLength / sentenceNum, 2)
        averageSentenceWrongNum = round(allWrongCharNum / sentenceNum, 2)
        averageNumEveryWrong = round(totalSentencesLength / allWrongCharNum, 2)
        

        content_list[0].append(file)
        content_list[1].append(sentenceNum)
        content_list[2].append(rightSentenceNum)
        content_list[3].append(wrongSentenceNum)
        content_list[4].append(maxSentencelength)
        content_list[5].append(minSentencelength)
        content_list[6].append(averageSentenceLength)
        content_list[7].append(allWrongCharNum)
        content_list[8].append(averageSentenceWrongNum)
        content_list[9].append(averageNumEveryWrong)
        content_list[10].append(taWrongSentenceNum)
        content_list[11].append(deWrongSentenceNum)
        content_list[12].append(taWrongCharNum)
        content_list[13].append(deWrongCharNum)

    save_to_csv(header_list, content_list, output_path)




def judgeWrongType(src,tgt,wrongTypeList):
    '''
    param:
        src: sentences containing errors
        tgt: modified target sentence
        wrongTypeList: specific error character group
    function:
        determine whether the sentence contains errors in a specific character group
    '''
    srcList = list(src)
    tgtList = list(tgt)
    sentenceTypeWrong = 0
    for ch1, ch2 in zip(srcList,tgtList):
        if ch1 != ch2 and ch1 in wrongTypeList:
            sentenceTypeWrong += 1
    if(sentenceTypeWrong != 0):
        return True, sentenceTypeWrong
    else:
        return False, 0
            

getSurfaceFeature("dataset/testDataset/","datasetSurfaceFeatureAnalysis.csv")

    
