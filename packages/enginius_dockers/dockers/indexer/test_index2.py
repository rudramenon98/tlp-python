import json
import concurrent.futures
import requests
import time

name = f'scann'

# dispatch requests in parallel
batch_size = 1
url = f'http://localhost:21501/search'

paraphrase = {"num_results": 100,
               "queries": ["flightcrew duty period", "pilot duty timings", "flightcrew flight duty period"]
              }

paraphrase = {"seq_0": "flightcrew duty period", "seq_1": 100}
with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
    def worker_thread(worker_index):
        # we'll send half the requests as not_paraphrase examples for sanity
        data = paraphrase #if worker_index < batch_size//2 else not_paraphrase
        print('Input: ', worker_index, data)
        t0 = time.time()
        response = requests.post(url, json=data)
        t1 = time.time()
        print('Status: ', response.status_code)
        print('Output: ', worker_index, response.json())

        print(f"time taken = {t1-t0}")

    for worker_index in range(batch_size):
        executor.submit(worker_thread, worker_index)

