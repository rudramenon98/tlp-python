import requests
from common import url


def test_api():
    # get all the scrapers
    response = requests.get(f"{url}/scrapers")
    print(f"Scrapers: {response.json()}")
    scraper_str = "EU_MDR_Ammendment"

    # run the scraper
    response = requests.post(
        f"{url}/run", json={"scraper": scraper_str, "scrapeURLId": 1}
    )
    print(f"Response: {response.json()}")


if __name__ == "__main__":
    test_api()
