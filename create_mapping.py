# # print (id_to_tweet_map[1454])
# # print("Keys in tweet_to_id_map:", list(tweet_to_id_map.keys()))
# # print (tweet_to_id_map['hasoc_hi_1059'])
# # print (tweet_to_id_map['Live aane ke liye switch to #JIO services !!!'])
# # print (id_to_class_map[4665])

# # this code is working perfecrly.

import csv

text_id = 2
id_to_tweet_map = {}
tweet_to_id_map = {}
id_to_class_map = {}

try:
    with open('hate_speech.tsv', encoding='utf-8') as dataset:
        for line_number, line in enumerate(csv.reader(dataset, delimiter="\t"), start=1):
            class_name = []
            text = line[0]

            if len(line) > 1 and len(line[1]) > 0:
                class_name.append(line[1])

            id_to_tweet_map[text_id] = text
            tweet_to_id_map[text] = text_id
            id_to_class_map[text_id] = class_name

            text_id += 1

except UnicodeDecodeError as e:
    print(f"Error decoding line {line_number}: {e}")
    # Handle or investigate the issue with the problematic line




#print id_to_tweet_map[1454]
#print tweet_to_id_map['Live aane ke liye switch to #JIO services !!!']
#print id_to_class_map[1454]

