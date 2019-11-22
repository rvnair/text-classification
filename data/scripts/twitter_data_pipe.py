import csv
import random as rand
inpath = "../data/twitter_in.csv"
outpath = "data/twitter/twitter_train.csv"
outpath_test = "data/twitter/twitter_test.csv"
outpath_valid = "data/twitter/twitter_valid.csv"

with open(inpath, 'r', encoding='mac_roman') as csv_file:
    with open(outpath, 'w') as csv_out, open(outpath_valid, 'w') as csv_valid, open(outpath_test, 'w') as csv_test:
        csv_reader = list(csv.reader(csv_file, delimiter=','))

        csv_writer = csv.writer(csv_out, delimiter=',')
        csv_valid = csv.writer(csv_valid, delimiter=',')
        csv_test = csv.writer(csv_test, delimiter=',')

        line_count = 0
        header = ['tweet','sentiment']
        csv_writer.writerow(header)
        csv_valid.writerow(header)
        csv_test.writerow(header)

        rands1 = rand.sample(range(800000), 25000)
        rands2 = rand.sample(range(800000, 1600000), 25000)

        for i,r in enumerate(rands1):
            row = csv_reader[r]
            if i < 2500:
                csv_test.writerow([csv_reader[r][5], csv_reader[r][0]])
            elif 2500 <= i < 5000:
                csv_valid.writerow([csv_reader[r][5], csv_reader[r][0]])
            else:
                csv_writer.writerow([csv_reader[r][5], csv_reader[r][0]])
        for i,r in enumerate(rands2):
            row = csv_reader[r]
            if i < 2500:
                csv_test.writerow([csv_reader[r][5], csv_reader[r][0]])
            elif 2500 <= i < 5000:
                csv_valid.writerow([csv_reader[r][5], csv_reader[r][0]])
            else:
                csv_writer.writerow([csv_reader[r][5], csv_reader[r][0]])
