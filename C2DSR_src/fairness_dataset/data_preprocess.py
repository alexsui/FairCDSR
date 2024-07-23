import pandas as pd
import random
import glob 
import sys
def preprocess(folder, dataset_name):
    random.seed(2023)
    data1,data2 = folder.split("_")
    if 'sci-fi' == data1:
        data1 = 'Sci-Fi' 
    if 'sci-fi' == data2:
        data2 = 'Sci-Fi'
    if 'film-noir' == data1:
        data1 = 'Film-Noir'
    if 'film-noir' == data2:
        data2 = 'Film-Noir'
    A_data_name = data1.capitalize() if data1!="Sci-Fi" and data1!="Film-Noir" else data1
    B_data_name = data2.capitalize() if data2!="Sci-Fi" and data2!="Film-Noir" else data2
    folder_name = f"{dataset_name}/{data1.lower()}_{data2.lower()}"
    A_data_path = folder_name + "/" + A_data_name+".csv"
    B_data_path = folder_name + "/" + B_data_name+".csv"
    A_df = pd.read_csv(A_data_path)
    B_df = pd.read_csv(B_data_path)
    unique_item_A = set(A_df['iid'])
    unique_item_B = set(B_df['iid'])

    # 將兩個domain data分為non-overlapped與overlapped
    # Identifying the unique users in each dataset
    unique_users_A = set(A_df['uid'])
    unique_users_B = set(B_df['uid'])

    # Identifying overlapped and non-overlapped users
    overlapped_users = unique_users_A & unique_users_B
    non_overlapped_users_A = unique_users_A - overlapped_users
    non_overlapped_users_B = unique_users_B - overlapped_users
    # remapping the user IDs 
    unique_item_A = set(A_df['iid'])
    iid_map_A = {iid: i for i, iid in enumerate(sorted(unique_item_A))}
    max_iid_A = max(iid_map_A.values()) + 1
    unique_item_B = set(B_df['iid'])
    iid_map_B = {iid: i+max_iid_A for i, iid in enumerate(sorted(unique_item_B))}
    A_df['iid'] = A_df['iid'].map(iid_map_A)
    B_df['iid'] = B_df['iid'].map(iid_map_B)

    overlapped_A_df = A_df[A_df['uid'].isin(overlapped_users)]
    overlapped_B_df = B_df[B_df['uid'].isin(overlapped_users)]
    non_overlapped_A_df = A_df[A_df['uid'].isin(non_overlapped_users_A)]
    non_overlapped_B_df = B_df[B_df['uid'].isin(non_overlapped_users_B)]

    # Creating the respective datasets
    overlapped_df = pd.concat([
        overlapped_A_df,
        overlapped_B_df
    ])

    # Saving the datasets to CSV files
    overlapped_df.to_csv(f'{folder_name}/{A_data_name}_{B_data_name}_overlapped.csv', index=False)
    non_overlapped_A_df.to_csv(f'{folder_name}/non_overlapped_{A_data_name}.csv', index=False)
    non_overlapped_B_df.to_csv(f'{folder_name}/non_overlapped_{B_data_name}.csv', index=False)
        
    # 將overlapped user 切為train, valid, test data
    def data_prepare_for_overlapped(folder_name):
        def split_data(user_sequences):
            train_ratio, validation_ratio = 0.75, 0.10
            n_total = len(user_sequences)
            n_train = int(n_total * train_ratio)
            n_validation = int(n_total * validation_ratio)
            train_data = user_sequences.iloc[:n_train]
            validation_data = user_sequences.iloc[n_train:n_train + n_validation]
            test_data = user_sequences.iloc[n_train + n_validation:]
            return train_data, validation_data, test_data
        def generate_small_sequences_for_test(sequences, min_length=10):
            """ Generate small sequences of at least 'min_length' with possible overlaps """
            # length_range = list(range(min_length, 15))
            small_sequences = []
            for uid, items in sequences:
                start = 0
                while True:
                    length = random.randint(min_length,15)
                    nonoverlapped_length = int(length*0.7)
                    if start+length > len(items):
                        if start==0:
                            small_sequences.append((uid, items))
                        break
                    end = start + length
                    small_seq = items[start:end]
                    start = start + nonoverlapped_length
                    small_sequences.append((uid, small_seq))
            return small_sequences
        def generate_small_sequences(sequences, m_length=10):
            small_sequences = []
            for uid, items in sequences:
                c=0
                for i in range(5):
                    length = random.randint(m_length, 40)
                    idx = len(items) - length + 1
                    if idx<0:
                        c+=1
                        if c==5:
                            small_sequences.append((uid, items))
                        continue
                    start = random.randint(0,idx)
                    end = start + length
                    small_seq = items[start:end]
                    small_sequences.append((uid, small_seq))
            return small_sequences
        data_name = f'{folder_name}/{A_data_name}_{B_data_name}_overlapped.csv'
        overlapped_data  =  pd.read_csv(data_name)
        overlapped_data = overlapped_data.sort_values(['timestamp'])
        train_data, validation_data, test_data = split_data(overlapped_data)
        
        # Grouping by user ID and sorting by timestamp within each group
        train_data = train_data.sort_values(['uid', 'timestamp']).groupby('uid')
        validation_data = validation_data.sort_values(['uid', 'timestamp']).groupby('uid')
        test_data = test_data.sort_values(['uid', 'timestamp']).groupby('uid')
        train_data = {uid: group[['iid','timestamp', 'gender']].values.tolist() for uid, group in train_data}
        validation_data  = {uid: group[['iid','timestamp', 'gender']].values.tolist() for uid, group in validation_data}
        test_data  = {uid: group[['iid','timestamp', 'gender']].values.tolist() for uid, group in test_data}
        train_data = [(k,v) for k,v in train_data.items()]
        validation_data = [(k,v) for k,v in validation_data.items()]
        test_data = [(k,v) for k,v in test_data.items()]
        train_data = generate_small_sequences(train_data)
        validation_data = generate_small_sequences(validation_data)
        test_data = generate_small_sequences_for_test(test_data)
        random.shuffle(train_data)
        random.shuffle(validation_data)
        random.shuffle(test_data)
        new_train_data = []
        length_threshold = 10
        for data in train_data:
            uid = data[0]
            gender = data[1][0][-1]
            seq = [d[0:2]for d in data[1]] # add timestamp
            if len(seq)<length_threshold:
                if gender==1 or (len(seq)<5 and gender==0):
                    continue    
            new_train_data.append((uid,gender,seq))
        new_valid_data = []
        for data in validation_data:
            uid = data[0]
            gender = data[1][0][-1]
            seq = [d[0:2]for d in data[1]]
            if len(seq)<length_threshold:
                continue
            new_valid_data.append((uid,gender,seq))
        new_test_data = []
        for data in test_data:
            uid = data[0]
            gender = data[1][0][-1]
            seq = [d[0:2]for d in data[1]]
            if len(seq)<length_threshold:
                continue
            new_test_data.append((uid,gender,seq))
        A_df = pd.read_csv(A_data_path)
        B_df = pd.read_csv(B_data_path)
        number_of_A_item, number_of_B_item = A_df.iid.nunique(), B_df.iid.nunique()
        def filter(sequences):
            for seq in sequences:
                if all([i<number_of_A_item for i,t in seq[2]]) or all([i>=number_of_A_item for i,t in seq[2]]):
                    sequences.remove(seq)
            return sequences
        new_train_data = filter(new_train_data)   
        new_valid_data = filter(new_valid_data)
        new_test_data = filter(new_test_data)
        target_test_len =int(0.15*(len(new_train_data)+len(new_valid_data))/0.85)
        new_test_data = new_test_data[:target_test_len]
        # balanced test data ratio as train data
        ratio = len([x for x in new_train_data if x[1]==1])/len([x for x in new_train_data if x[1]==0])
        print(f"Overlapped domain: {A_data_name} & {B_data_name}\n")
        print("Overlapped user:\n")
        print(f"train_data size:{len(new_train_data)}, ratio:{len(new_train_data)/(len(new_train_data)+len(new_valid_data)+len(new_test_data))}")
        print(f"valid_data size:{len(new_valid_data)},ratio:{len(new_valid_data)/(len(new_train_data)+len(new_valid_data)+len(new_test_data))}")
        print(f"test_data size:{len(new_test_data)},ratio:{len(new_test_data)/(len(new_train_data)+len(new_valid_data)+len(new_test_data))}")
        print(f"train_data male/female ratio: ", ratio)
        print(f"number of test data male :",len([x for x in new_test_data if x[1]==1]))
        print(f"number of test data female :",len([x for x in new_test_data if x[1]==0]),"\n")
        return new_train_data,new_valid_data,new_test_data, number_of_A_item,number_of_B_item
    overlapped_train_data, overlapped_valid_data, overlapped_test_data, number_of_A_item, number_of_B_item = data_prepare_for_overlapped(folder_name)
    def save_sequences_to_txt(file_path, sequences, num_items_x, num_items_y):
        with open(file_path, 'w') as file:
            # Writing the number of items in X and Y domains
            file.write(f"{num_items_x}\n")
            file.write(f"{num_items_y}\n")
            
            for uid, gender, seq in sequences:
                seq = [str(item[0])+"|"+str(item[1]) for item in seq]
                data =  [uid, gender] +seq
                line = ' '.join(map(str,data))
                file.write(line + "\n")            
    save_sequences_to_txt(f'{folder_name}/train.txt', overlapped_train_data, number_of_A_item, number_of_B_item)
    save_sequences_to_txt(f'{folder_name}/valid.txt', overlapped_valid_data, number_of_A_item, number_of_B_item)
    save_sequences_to_txt(f'{folder_name}/test.txt', overlapped_test_data, number_of_A_item, number_of_B_item)
        
if __name__ == '__main__':
    dataset_name = sys.argv[1]    
    folder_list = glob.glob(f"./{dataset_name}/*")
    folder_list = [x.split("/")[-1] for x in folder_list]
    print(folder_list)
    for folder in folder_list:
        try:
            print("*"*40)
            print(f"Preprocessing {folder} ...")    
            preprocess(folder,dataset_name)
            print(f"Finish {folder}\n")
            print("*"*40)
        except Exception as e:
            print(e)
            continue
        