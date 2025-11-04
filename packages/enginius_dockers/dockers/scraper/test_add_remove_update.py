import requests
from common import url


class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def add_texts(self, ids, texts):
        """
        Add vectors with IDs to the index.

        :param ids: List of int IDs
        :param vectors: List of vector lists (floats)
        :return: Response JSON
        """
        url = f"{self.base_url}/add"
        payload = {"ids": ids, "texts": texts}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def remove_ids(self, ids):
        """
        Remove vectors by IDs from the index.

        :param ids: List of int IDs to remove
        :return: Response JSON
        """
        url = f"{self.base_url}/remove"
        payload = {"ids": ids}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def update(self, update_data):
        try:
            url = f"{self.base_url}/update"
            response = requests.post(url, json=update_data)
            response.raise_for_status()
            result = response.json()
            return result
            print("✅ Successfully updated index:")
            print(result)
        except requests.exceptions.RequestException as e:
            print("❌ Request failed:")
            print(e)
            if e.response is not None:
                print(e.response.text)


# Example usage:

if __name__ == "__main__":
    client = APIClient(url)

    # Add texts
    ids_to_add = [11111, 11112, 11113, 11114]
    texts_to_add = [
        "flightcrew duty period",
        "flightcrew flight duty period",
        "flight crew duty hours maximum",
        "flightcrew duty period rules",
    ]
    print("Adding vectors...")
    add_resp = client.add_texts(ids_to_add, texts_to_add)
    print(add_resp)

    # Remove ids
    ids_to_remove = [11112]
    print("Removing ids...")
    remove_resp = client.remove_ids(ids_to_remove)
    print(remove_resp)

    update_texts = {
        "updates": {
            "1": "This is a test document.",
            "2": "Another text that should be indexed.",
            "3": "FastAPI makes it easy to create APIs.",
            "4": "FAISS is great for vector search.",
            "5": "This one should replace existing ID 1 if already present.",
        }
    }

    print("Updating...")
    update_resp = client.update(update_texts)
    print(update_resp)
