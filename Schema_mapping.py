#pip install schema-matching
import os
import pandas as pd
# from schema_matching import schema_matching
import numpy as np
import locale
from locale import atof
import warnings
import itertools
import collections
import pathlib
import csv

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

fileExt_1 = r"similarity_matrix_value.csv"
fileExt_2 = r"table2.csv"
matrix_path_name_pairs = []

cwd = [x[0] for x in os.walk(os.getcwd())]
subdir = [os.path.basename(x) for x in cwd]
company_names = []
for dir_name in cwd:
    company_names.append(dir_name.split('/')[-1].split('_')[-2])
repeated_company_names = [item for item, count in collections.Counter(company_names).items() if count > 1]
repeated_company_names_copy = repeated_company_names.copy()

repeated_company_paths = []
for subdir, dirs, files in os.walk('./'):
    for company_name in repeated_company_names:
            if company_name in subdir:
                if subdir.split('/')[-1].split('_')[-2] + '_' + subdir.split('/')[-1].split('_')[-1] not in repeated_company_paths:
                    repeated_company_paths.append(os.getcwd() + subdir[1:])
# add new
non_repeat_path = []
for subdir, dirs, files in os.walk('./'):
    if subdir.split('/')[-1] != '':
        if subdir.split('/')[-1].split('_')[-2] not in repeated_company_names:
            non_repeat_path.append(os.getcwd() + subdir[1:])

tobe_merged = []
for path in repeated_company_paths:
    for file in os.listdir(path):
        if file.endswith(fileExt_2):
            tobe_merged.append((path, file))

non_repeat_merge = []
for path in non_repeat_path:
    for file in os.listdir(path):
        if file.endswith(fileExt_2):
            non_repeat_merge.append((path, file))

cur_path = os.getcwd()
os.chdir('..')
folder_name = pathlib.Path(os.getcwd(), 'Results')
folder_name.mkdir(parents=True, exist_ok=True)
result_path = os.getcwd() + '/Results'
os.chdir(cur_path)

concated = []
for path, file in tobe_merged:
    path_file_1 = path+'/'+file
    if path_file_1 not in concated:
        table2 = pd.read_csv(path_file_1)
        # start

        d2_path = os.path.join(path, 'similarity_matrix_value.csv')
        df2 = pd.read_csv(d2_path,index_col=0)

        columns = df2.columns.values
        
        guru_columns = df2.index.values  # list of gurufocus columns
        table2_columns = table2.columns
        
        df2_array = np.array(df2)
        
        temp = np.argmax(df2_array, axis = 1)
        
        for i in range(len(df2_array[0])):
            table2 = table2.rename(columns={table2_columns[i]: guru_columns[temp[i]]}, inplace=False)

        # end
        for path2, file2 in tobe_merged:
            path_file_2 = path2+'/'+file2
            if path_file_2 not in concated and path_file_2 != path_file_1:
                if path_file_1.split('/')[-2].split('_')[-2] == path_file_2.split('/')[-2].split('_')[-2]:
                    table2_2 = pd.read_csv(path_file_2)
                    #start

                    d2_path2 = os.path.join(path2, 'similarity_matrix_value.csv')
                    df2_2 = pd.read_csv(d2_path2,index_col=0)

                    columns_2 = df2_2.columns.values
                    
                    guru_columns_2 = df2_2.index.values  # list of gurufocus columns
                    table2_columns_2 = table2_2.columns
                    
                    df2_array_2 = np.array(df2_2)
                    
                    temp_2 = np.argmax(df2_array_2, axis = 1)
                    
                    for i in range(len(df2_array_2[0])):
                        table2_2 = table2_2.rename(columns={table2_columns_2[i]: guru_columns_2[temp[i]]}, inplace=False)
                    table2 = pd.concat([table2_2, table2], axis=1)
                    concated.append(path2+'/'+file2)
        concated.append(path+'/'+file)
        duplicate_header_mask = table2.columns.duplicated()
        duplicate_col_mask = table2.T.duplicated()
        overall_mask = np.bitwise_or(~duplicate_header_mask, ~duplicate_col_mask)
        table2 = table2.iloc[:,overall_mask.values]

        new_folder_name = path_file_1.split('/')[-2].split('_')[-3] + '_' + path_file_1.split('/')[-2].split('_')[-2]
        gurufocus_path = os.path.dirname(path_file_1) + '/' + 'gurufocus.csv'
        df = pd.read_csv(gurufocus_path)
        # concat the columns.values that are in gurufocus but not in table2
        for col in df.columns.values:
            if col not in table2.columns.values:
                table2[col] = np.nan
        os.chdir(result_path)
        new_folder_path = pathlib.Path(os.getcwd(), new_folder_name)
        table2.replace('_PO_', 0, inplace=True)
        table2.fillna(0, inplace=True)
        zero_mask = []
        for each in table2.columns.values:
            if each == "0":
                zero_mask.append(False)
            else:
                zero_mask.append(True)
        table2 = table2.iloc[:,zero_mask]
        table2 = table2.loc[:,~table2.columns.duplicated()].copy()
        table2.to_csv(new_folder_name + '.csv', index=False)
        os.chdir(cur_path)


for path, file in non_repeat_merge:
    path_file_1 = path+'/'+file
    new_folder_name = path_file_1.split('/')[-2].split('_')[-3] + '_' + path_file_1.split('/')[-2].split('_')[-2]
    gurufocus_path = os.path.dirname(path_file_1) + '/' + 'gurufocus.csv'
    df = pd.read_csv(gurufocus_path)
    # concat the columns.values that are in gurufocus but not in table2
    for col in df.columns.values:
        if col not in table2.columns.values:
            table2[col] = np.nan
    os.chdir(result_path)
    new_folder_path = pathlib.Path(os.getcwd(), new_folder_name)
    table2.replace('_PO_', 0, inplace=True)
    table2.fillna(0, inplace=True)
    zero_mask = []
    for each in table2.columns.values:
        if each == "0":
            zero_mask.append(False)
        else:
            zero_mask.append(True)
    table2 = table2.iloc[:,zero_mask]
    table2 = table2.loc[:,~table2.columns.duplicated()].copy()
    table2.to_csv(new_folder_name + '.csv', index=False)
    os.chdir(cur_path)

# # commented out
# for subdir, dirs, files in os.walk('./'):

#     for filename in files:
#         if filename.endswith(fileExt_1):
#             matrix_path_name_pairs.append((subdir,filename))

# counter = 1
# for i in matrix_path_name_pairs:
#     # print("Processing file: ", counter)
#     counter += 1
#     # print(i)
#     try:
#         (subdir,filename) = i
#         d2_path = os.path.join(subdir, filename)
#         df2 = pd.read_csv(d2_path,index_col=0)
    
#         table2_address =  subdir + "/table2.csv"
        
#         table2 = pd.read_csv(table2_address)
#         columns = df2.columns.values
        
#         guru_columns = df2.index.values  # list of gurufocus columns
#         table2_columns = table2.columns


        
#         out_filename = "insider_result/" + subdir.split('/')[-1] + ".csv"
        
        
        
#         df2_array = np.array(df2)
        
#         temp = np.argmax(df2_array, axis = 1)
        
#         for i in range(len(df2_array[0])):
            
#             if df2.iloc[temp[i], i] >= 0.4:
#                 table2 = table2.rename(columns={table2_columns[i]: guru_columns[temp[i]]}, inplace=False)
#             else:
#                 # print(out_filename + ' : ' + table2_columns[i] )
#                 pass
            
        
        
#         # print('--------------')
        
        
        
        
        
        
#         table2_df = pd.DataFrame(table2)


#         # print(table2.columns.values)

#         # drop columns that has the same value and same row header
#         duplicate_header_mask = table2_df.columns.duplicated()
#         duplicate_col_mask = table2_df.T.duplicated()
#         # print(duplicate_col_mask)
#         # print(duplicate_mask)
#         overall_mask = np.bitwise_or(~duplicate_header_mask, ~duplicate_col_mask)
#         test = table2_df.iloc[:,overall_mask.values]
#         test = test.where(test != '_PO_', 0)
#         to_be_appended = []
#         for column in guru_columns:
#             if column not in test.columns:
#                 to_be_appended.append(column)
#         if(to_be_appended):
#             test[to_be_appended] = 0
#             last_row = test.index[-1]
#             if(last_row == 'TTM'):
#                 current_year = str(int(test.index[-2])+1)
#                 test.rename(index={'TTM': current_year}, inplace=True)
#         region = subdir.split('/')[-1].split('_')[-3]
#         # print(region)
#         code = subdir.split('/')[-1].split('_')[-2]
#         # print(code)
#         # print('--------------')
#         test.insert(loc=0, column='code', value=code)
#         test.insert(loc=0, column='region', value=region)
#         # print(out_filename)
#         if '13f' in out_filename:
#             test.to_csv(out_filename, index= False)
#             print("reached saving file")
#             os.rmdir(subdir)
#     except:
#         pass
#     #    print("error:")
#     #    print(filename)

# def merge_csv(fileList, newFileName):
#     allHeaders = set([])
#     for afile in fileList:
#         with open(afile, 'rb') as csvfilesin:
#             eachheader = csv.reader(csvfilesin, delimiter=',').next()
#             allHeaders.update(eachheader)
#     print(allHeaders)
#     with open(newFileName, 'wb') as csvfileout:
#         outfile = csv.DictWriter(csvfileout, allHeaders)
#         outfile.writeheader()
#         for afile in fileList:
#             print('***'+afile)
#             with open(afile, 'rb') as csvfilesin:
#                 rows = csv.DictReader(csvfilesin, delimiter=',')
#                 for r in rows:
#                     print(allHeaders.issuperset(r.keys()))
#                     outfile.writerow(r)