import os
import pandas as pd
import random


def convert_labeled_list(root, csv_list):
    img_list = list()
    label_list = list()
    for csv_file in csv_list:
        data = pd.read_csv(os.path.join(root, csv_file))
        img_list += data['image'].tolist()
        label_list += data['mask'].tolist()

    # seed = 42
    # random.seed(seed)
    # random.shuffle(img_list)
    # random.seed(seed)
    # random.shuffle(label_list)
    return img_list, label_list
