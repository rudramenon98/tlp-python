import requests

# Config
url = "http://localhost:11000"  # Update if hosted remotely
doc_ids = [100]
pdf_type = "Arxiv"


def call_inference_api(doc_ids: list[int]):
    data = {"doc_ids": doc_ids, "pdf_type": pdf_type}
    response = requests.post(f"{url}/predict", data=data)

    if response.status_code == 200:
        print("✅ Prediction Results:")
        results = response.json()
        for item in results:
            print(f"Status: {item.get('status', 'unknown')}")
            if "processed_docs" in item:
                print(f"Processed docs: {item['processed_docs']}")
            if "failed_docs" in item:
                print(f"Failed docs: {item['failed_docs']}")
            print(f"Message: {item.get('message', 'No message')}")
            print("---")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

    # Call summarize api
    data = {"doc_ids": doc_ids}
    response = requests.post(f"{url}/summarize", data=data)
    print(response.json())


if __name__ == "__main__":
    doc_ids = [100]
    call_inference_api(doc_ids)
