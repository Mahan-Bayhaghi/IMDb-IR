import json
import LSH
import time


def import_all_summaries(filepath, restrict=False, restrict_num=200):
    with open(filepath, 'r') as file:
        data = json.load(file)
    temp_summaries = [movie['summaries'] for movie in data]
    if restrict:
        temp_summaries = temp_summaries[:restrict_num]
    summaries = []
    for summary in temp_summaries:
        long_summary = ""
        for item in summary:
            long_summary += item
            long_summary += " "
        summaries.append(long_summary)
    return summaries


all_summaries = import_all_summaries("../IMDB_crawled.json", restrict=True, restrict_num=200)
all_summaries += import_all_summaries("./LSHFakeData.json")

minhash_lsh = LSH.MinHashLSH(all_summaries, num_hashes=100)
t = time.time()
buckets = minhash_lsh.perform_lsh()
t = time.time() - t
minhash_lsh.jaccard_similarity_test(buckets, all_summaries)
print(f"elapsed time: {t} seconds")
