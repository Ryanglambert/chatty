from functools import partial
from pyspark import SparkContext

import tokens



if __name__ == "__main__":
    text_path = 's3://daily-dialogue/dialogues_text.txt'
    with open(text_path, 'r') as f:
        conversations = f.readlines()

    tokenizer = partial(tokens.tokenize_as_list, sep='_', tokenizers=[tokens.lemma])
    sc = SparkContext()
    par_conversations = sc.parallelize(conversations)
    counts = par_conversations.map(tokenizer)\
                              .saveAsTextFile('s3://daily-dialogue/result')
    sc.stop()
