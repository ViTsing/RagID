# Calculate the overlap rate of the 100 most frequently occurring incorrect character pairs in different data sets
# (1).Overlap rate = (intersection number of error correction characters in two data sets) / (union number of error correction characters in two data sets)
# (2).Overlap rate = (intersection number of error correction characters in the first 100 of the two data sets) / (100)
# 1. Get the data and dataset names of each dataset from pkl file
# 2. Get the mapping of all incorrect characters to correct characters in each data set, sorted in descending order of number of occurrences
# 3. Get the mapping set of the first N wrong characters to the correct characters for each data set, and calculate the set repetition rate of different data sets

from tools.fileOperationTools import readPklFile,getDictDatasetFromPklDataset, getFilesFromFolder,save_to_csv


def getCharMappingDictFromDataset(dict_dataset):
    '''
    param:
        strList1: List of strings (input data)
        strList2: List of strings (output data)
    function:
        Gives the mapping pairs and occurrence counts of all incorrect characters to correct characters in a dataset
    '''
    # A dictionary that records the mapping of all errors to correct characters and the number of occurrences
    char_map_dict = {} 

    # Ensure both lists are of the same length
    input_list = dict_dataset.get("inputList")
    output_list = dict_dataset.get("outputList")
    for str1, str2 in zip(input_list, output_list):
        if(len(str1) == len(str2)):
        # Compare characters of both strings
            for char1, char2 in zip(str1, str2):
                if char1 != char2:
                    key = f"{char1}_{char2}"  # char map: char1_char2
                    char_map_dict[key] = char_map_dict.get(key, 0) + 1  # Number of occurrences

    return char_map_dict

def getTopNOccurrenceOfCharMapping(char_map_dict,N):
    '''
    param:
        char_map_dict: char mapping dict from function 'getCharMappingDictFromDataset'
        N: Get the first N char_maps
    function:
        Get a list of character maps contains N elements
    '''
    # Sort by occurrence in descending order
    sorted_char_mapping_list = sorted(char_map_dict, key=char_map_dict.get, reverse=True)
    return sorted_char_mapping_list[:N]

def calculateErrorOverlapBetweenTwoList(sorted_char_mapping_list_1,sorted_char_mapping_list_2):
    '''
    param:
        sorted_char_mapping_list_1: List of elements (first list)
        sorted_char_mapping_list_2: List of elements (second list)
        
    function:

    '''
    # Convert both lists to sets and find the intersection
    set1, set2 = set(sorted_char_mapping_list_1), set(sorted_char_mapping_list_2)
    intersection = set1.intersection(set2)
    # Calculate the number of common elements
    n = len(intersection)
    # Calculate the overlap rate (intersection size / total size of the lists)
    overlap_rate = n / max(len(sorted_char_mapping_list_1), len(sorted_char_mapping_list_2))  # Ensure overlap rate is based on the larger list size
    temp_dict = {
        "overlap_num": n,
        "overlap_rate": overlap_rate
    }
    return temp_dict

def caculateOverlapAmongDataset(dataset_path_list):
    '''
    param:
        dataset_path_list: List of file paths for the dataset in pkl format.
        
    function:
        Load all datasets, extract data and names, and calculate overlap rate between each pair of datasets.
    '''
    # Store the datasets and their names
    dataset_data_list = []
    dataset_name_list = []
    
    # Load each dataset from the given paths
    for pkl_path in dataset_path_list:
        data = readPklFile(pkl_path)
        dataset_data_list.append(data)
        dataset_name_list.append(pkl_path.split('/')[-1])  # Assuming the file name is a good identifier

    overlap_results = {}

    # Compare each pair of datasets (including each dataset with itself)
    for i in range(len(dataset_name_list)):
        for j in range(i, len(dataset_name_list)):  # Ensure the pair is compared once (including i == j)
            dataset1 = dataset_data_list[i]
            dataset2 = dataset_data_list[j]
            name1 = dataset_name_list[i]
            name2 = dataset_name_list[j]

            structed_Dataset1 = getDictDatasetFromPklDataset(dataset1)
            structed_Dataset2 = getDictDatasetFromPklDataset(dataset2)
            char_map_list1 = getTopNOccurrenceOfCharMapping(getCharMappingDictFromDataset(structed_Dataset1),100)
            char_map_list2 = getTopNOccurrenceOfCharMapping(getCharMappingDictFromDataset(structed_Dataset2),100)
            error_overlap_value = calculateErrorOverlapBetweenTwoList(char_map_list1,char_map_list2)
            overlap_results[name1 + "+" + name2] = error_overlap_value
    return overlap_results

if __name__ == "__main__":
    file_list = getFilesFromFolder("dataset/testDataset/")
    file_list = [file for file in file_list if file.endswith('.pkl')]
    overlap_data = caculateOverlapAmongDataset(file_list)
    print(overlap_data)
    output_csv_path = "datasetOverlapAnalysis.csv"
    header = ['Key', 'Value']  # The header of the CSV file
    content = [[key for key, value in overlap_data.items()] ,[value for key, value in overlap_data.items()]]# Convert dict to list of tuples
    save_to_csv(header, content, output_csv_path)


 

   