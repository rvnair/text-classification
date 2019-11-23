import csv
import random as rand
inpath_train = "data/twitter/twitter_train.csv"
inpath_valid = "data/twitter/twitter_valid.csv"

import numpy as np
from scipy import stats

def noise_mtx(s, n_classes):
    return (1.0-s) * np.identity(n_classes) + float(s / n_classes)

def gen_rand_label(noise, label):
    dist = noise[label]
    rv = stats.rv_discrete(values=np.array(list(enumerate(dist))).T)
    return rv.rvs(size=1)[0]


noises = [.10,.20,.30,.40,.50,.60,.70,.80,.90,1.00]
with open(inpath_train, 'r', encoding='mac_roman') as csv_train, open(inpath_valid, 'r', encoding='mac_roman') as csv_valid:
    train_reader = list(csv.reader(csv_train, delimiter=','))
    valid_reader = list(csv.reader(csv_valid, delimiter=','))
    for n in noises:
        outpath_train = "data/twitter/rand/twitter_{}_train.csv".format(int(n * 100))
        outpath_valid = "data/twitter/rand/twitter_{}_valid.csv".format(int(n * 100))

        noise = noise_mtx(n, 2)

        with open(outpath_train, 'w') as csv_train_w, open(outpath_valid, 'w') as csv_valid_w:
            write_train = csv.writer(csv_train_w, delimiter=',')
            write_valid = csv.writer(csv_valid_w, delimiter=',')

            line_counter = 0
            for row in train_reader:                
                label_id = 0 if row[1] == "0" else 1
                label_id_wrt = gen_rand_label(noise, label_id)
                label_wrt = "0" if label_id_wrt == 0 else "4"
                out = [row[0], label_wrt]

                line_counter += 1
                write_train.writerow(out)
            
            line_counter = 0
            for row in valid_reader:
                label_id = 0 if row[1] == "0" else 1
                label_id_wrt = gen_rand_label(noise, label_id)
                label_wrt = "0" if label_id_wrt == 0 else "4"
                out = [row[0], label_wrt]
                
                line_counter += 1
                write_valid.writerow(out)