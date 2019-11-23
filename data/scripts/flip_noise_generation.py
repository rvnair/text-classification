import csv
import random as rand
inpath_train = "data/twitter/twitter_train.csv"
inpath_valid = "data/twitter/twitter_valid.csv"

noises = [10,20,30,40,50,60,70,80,90,100]
with open(inpath_train, 'r', encoding='mac_roman') as csv_train, open(inpath_valid, 'r', encoding='mac_roman') as csv_valid:
    train_reader = list(csv.reader(csv_train, delimiter=','))
    valid_reader = list(csv.reader(csv_valid, delimiter=','))
    for n in noises:
        outpath_train = "data/twitter/twitter_{}_train.csv".format(n)
        outpath_valid = "data/twitter/twitter_{}_valid.csv".format(n)

        with open(outpath_train, 'w') as csv_train_w, open(outpath_valid, 'w') as csv_valid_w:
            write_train = csv.writer(csv_train_w, delimiter=',')
            write_valid = csv.writer(csv_valid_w, delimiter=',')

            rands1 = range(40000) if n == 100 else rand.sample(range(40000), int(n * 40000 / 100))
            rands2 = range(5000) if n == 100 else rand.sample(range(5000), int(n * 5000 / 100))

            line_counter = 0
            for row in train_reader:
                out = []
                if (line_counter - 1) in rands1:
                    label = "0" if row[1] == "4" else "4"
                    out = [row[0], label]
                else:
                    out = row
                line_counter += 1
                write_train.writerow(out)
            
            line_counter = 0
            for row in valid_reader:
                out = []
                if (line_counter - 1) in rands2:
                    label = "0" if row[1] == "4" else "4"
                    out = [row[0], label]
                else:
                    out = row
                line_counter += 1
                write_valid.writerow(out)