import os
import pandas as pd

from svm import preprocess

# Output/test/AAL_statistics_volumn_test.csv
# Output/train/AAL_statistics_volumn_train.csv

def get_data(train_test="train", voxel=False):
    data_x, _, names = preprocess(f"Output/{train_test}/AAL_statistics_volumn_{train_test}.csv",-1 if voxel == True else 0)
    collated = {}
    for idx, x in enumerate(data_x):
        collated[names[idx]] = x
    data_return = pd.DataFrame(collated,index=[f"ROI_{x}" for x in range(len(collated[names[0]]))]).transpose()
    data_return.index.name="Filename"
    return data_return

if __name__ == "__main__":
    train_df = get_data("train",False)
    test_df = get_data("test",False)
    all_collated = pd.concat([train_df,test_df])
    all_collated.to_csv("overall_results.csv")
