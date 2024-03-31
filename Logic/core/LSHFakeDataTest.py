import json
import time
import LSH


def import_all_summaries(filepath, restrict=False, restrict_num=200):
    with open(filepath, 'r') as file:
        data = json.load(file)
    all_movies = [movie for movie in data]
    temp_summaries = [movie["summaries"] for movie in all_movies]
    if restrict:
        temp_summaries = temp_summaries[:restrict_num]
    summaries = [' '.join(summary) for summary in temp_summaries]
    return summaries


all_summaries = import_all_summaries("./LSHFakeData_preprocessed.json")
all_summaries += import_all_summaries("../IMDB_crawled_preprocessed.json", restrict=True, restrict_num=300)

minhash_lsh = LSH.MinHashLSH(all_summaries, num_hashes=800)
t = time.time()
buckets = minhash_lsh.perform_lsh(num_bands=100, num_rows_per_band=None)
t = time.time() - t
minhash_lsh.jaccard_similarity_test(buckets, all_summaries)
print(f"elapsed time for LSH: {t} seconds")
