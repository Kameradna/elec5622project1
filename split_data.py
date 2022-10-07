import os
import pandas as pd

def split_data():
    main_path = os.path.abspath(os.path.dirname(__file__))
    with open('Original/data_labels.csv') as f:
        label_map = pd.read_csv(f,dtype={'No':int,'File Name':str,'Label':str})
    labelled = label_map[label_map['Label'].notna()]
    unlabelled = label_map[label_map['Label'].isna()]
    # train_set, validation_set = model_selection.train_test_split(label_map, test_size=None, train_size=ratio, random_state=None, shuffle=True, stratify=label_map['Label'])
    # files = [f for f in os.listdir('Original') if f.endswith(".nii")]
    for label, content in labelled.iterrows():
        label = content['Label'].strip(' ')
        filename = content['File Name'].strip(' ')
        new_name = f'{label}_{filename}'
        if os.path.exists(f'{main_path}/Original/{filename}'):
            os.replace(f'{main_path}/Original/{filename}',f'{main_path}/Data/train/{new_name}')
    for label, content in unlabelled.iterrows():
        filename = content['File Name'].strip(' ')
        if os.path.exists(f'{main_path}/Original/{filename}'):
            os.replace(f'{main_path}/Original/{filename}',f'{main_path}/Data/test/{filename}')

if __name__ == "__main__":
    split_data()