import os
import pandas as pd

df_train = pd.DataFrame(columns=['class', 'text'])
df_test = pd.DataFrame(columns=['class', 'text'])
path = '/home/lguarise/Desktop/News_corpus/full_texts/'

for directory in os.listdir(path):
    directory_path = os.path.join(path, directory)
    if os.path.isdir(directory_path):
        count = 0
        if 'fake' in directory:
            TorF = 'fake'
        else:
            TorF = 'true'
        for filename in os.listdir(directory_path):
            with open(os.path.join(directory_path, filename), 'r') as f:
                text = f.read()
                current_df = pd.DataFrame([[TorF, text]],columns=['class', 'text'])
                count += 1
                if count%10 != 0:
                    df_train = df_train.append(current_df, ignore_index=True)
                else:
                    df_test = df_test.append(current_df, ignore_index=True)