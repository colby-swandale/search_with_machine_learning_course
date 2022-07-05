import fasttext
import csv

model_file      = r'/workspace/datasets/fasttext/title_model.bin'
top_words_file  = r'/workspace/datasets/fasttext/top_words.txt'
output_file     = r'/workspace/datasets/fasttext/synonyms.csv'

synonyms_model = fasttext.load_model(model_file)

input = open(top_words_file, 'r')
with open(output_file, 'w+', newline ='') as f:
    writer = csv.writer(f)
    rows   = []
    for line in input.readlines():
        word = line.strip()
        neighbours = filter(lambda x: x[0] >= 0.8, synonyms_model.get_nearest_neighbors(word))
        words = list(map(lambda x: x[1], neighbours))

        words.insert(0, word)
        rows.append(words)

    writer.writerows(rows)