import requests

API_URL = "http://localhost:8000/extract-terms"

# For testing inside certain docker setups, you might target the service name instead:
# API_URL = "http://dictionary:8000/extract-terms"

DEFAULT_DOCUMENT_IDS = [101, 202, 303]
DEFAULT_DOCUMENT_TYPES = [1, 2]


def call_api(document_ids=None, document_types=None):
    payload = {}
    if document_ids is not None:
        payload["document_ids"] = document_ids
    if document_types is not None:
        payload["document_types"] = document_types

    response = requests.post(API_URL, json=payload, timeout=5)

    if response.status_code == 200:
        print("✅ Response:")
        print(response.json())
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    call_api(DEFAULT_DOCUMENT_IDS, DEFAULT_DOCUMENT_TYPES)


