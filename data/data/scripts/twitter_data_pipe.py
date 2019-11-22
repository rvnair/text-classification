import csv
import random as rand
inpath = "data/twitter/twitter_in.csv"
outpath = "data/twitter/twitter.csv"
with open(inpath, 'r', encoding='mac_roman') as csv_file:
    with open(outpath, 'w') as csv_out:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        csv_writer = csv.writer(csv_out, delimiter=',')
        line_count = 0
        header = ['tweet','sentiment']
        csv_writer.writerow(header)
        for i in range(25000):
            idx = rand.randint(0, 799999)
            row = csv_reader[idx]
            csv_writer.writerow([csv_reader[idx][5], csv_reader[idx][0]])
        for i in range(25000):
            idx = rand.randint(800000, 1599999)
            row = csv_reader[idx]
            csv_writer.writerow([csv_reader[idx][5], csv_reader[idx][0]])
            # if line_count != 0:
            #     review = row[0].lower().split(' ')
            #     samples.append(review[:self.seqLen])
            #     labels.append(row[1])
            line_count += 1
