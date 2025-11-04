import concurrent.futures

import requests

# with open('config.json') as fp:
#    config = json.load(fp)
max_length = 128  # config['max_length']
batch_size = 1  # config['batch_size']
mname = f"bert-max_length{max_length}-batch_size{batch_size}"
name = f"dictionary"

# dispatch requests in parallel
url = f"http://localhost:29295/predictions/{name}"
paraphrase = {"seq_0": "FAA", "seq_1": 1}
#        "The company HuggingFace is based in New Jersey Cities"]
# not_paraphrase = {'seq_0': paraphrase['seq_0'], 'seq_1': 'This is total nonsense.'}

with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:

    def worker_thread(worker_index):
        # we'll send half the requests as not_paraphrase examples for sanity
        data = paraphrase  # if worker_index < batch_size//2 else not_paraphrase
        print("Input: ", worker_index, data)
        response = requests.post(url, data=data)
        print("Status: ", response.status_code)
        print("Output: ", worker_index, response.json())

    for worker_index in range(batch_size):
        executor.submit(worker_thread, worker_index)
