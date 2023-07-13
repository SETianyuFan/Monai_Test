import os, zipfile
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="你的程序描述")

    parser.add_argument('--arg1', type=int, default=0, help='arg1 的描述')
    parser.add_argument('--arg2', type=str, default='default', help='arg2 的描述')

    args = parser.parse_args()
    return args

def feature_generate(path_image, path_mask):

    params = os.path.join(os.getcwd(), '..', 'examples', 'exampleSettings', 'Params.yaml')
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    features = {}
    for case_id in range(1, 101):
        path_image = '/home/tianyu/Desktop/data_base/imagesTr_cropped4/data/'
        path_mask = '/home/tianyu/Desktop/data_base/imagesTr_cropped4/mask/'
        image = sitk.ReadImage(path_image + f"testdata{case_id}.nrrd")
        mask = sitk.ReadImage(path_mask + f"testdata{case_id}.nrrd")

        features[case_id] = extractor.execute(image, mask, label=1)
        print(f'case {case_id} done')

    # print(features[1].keys())
    # print(features[1].values())

    df = pd.DataFrame(features)

    import numpy as np

    # M label and B label graph feature
    dict1 = features[1]
    dict2 = features[2]

    different_values_keys = []

    for key1, value1 in dict1.items():
        if key1 in dict2 and dict2[key1] != value1:
            different_values_keys.append(key1)

    different_values_keys = np.array(different_values_keys)
    different_values_keys = different_values_keys[10:]

    # print(different_values_keys)
    # print(np.size(different_values_keys))



def main():
    args = parse_arguments()



if __name__ == "__main__":
    main()
