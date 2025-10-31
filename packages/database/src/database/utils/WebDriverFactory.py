import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as chromeOption


class WebDriverFactory:

    @staticmethod
    def getWebDriverInstance(browser, docker_url=None):

        file_path = os.path.abspath(os.path.join(__file__, "../../.."))
        chrome_driver_path = os.path.join(file_path, "chrome_driver/chromedriver")
        chrome_log = os.path.join(file_path, "chrome_driver/chrome_logs.log")

        if browser == "iexplorer":
            driver = webdriver.Ie()
        elif browser == "firefox":
            driver = webdriver.Firefox()
        elif browser == "chrome":
            chrome_options = chromeOption()
            # chrome_options.add_argument('user-data-dir={profile_dir}'.format(profile_dir="chrome_data/"))
            chrome_options.add_argument(
                "User-Agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
            )
            # self.chrome_options.add_argument("--disable-extensions")
            # self.chrome_options.add_argument("--disable-gpu")
            # self.chrome_options.add_argument("--no-sandbox") # linux only
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            # chrome_options.add_argument("--log-path=/home/hanuvadiya/TLP/tlp-scripts/chrome.log")
            chrome_options.add_argument("--headless")
            chrome_options.add_experimental_option(
                "excludeSwitches", ["enable-automation"]
            )
            chrome_options.add_experimental_option("useAutomationExtension", False)
            print("chrome driver path >> ", chrome_driver_path)
            driver = webdriver.Chrome(
                chrome_driver_path,
                options=chrome_options,
                service_args=["--verbose", "--log-path=" + chrome_log],
            )
        elif browser == "docker":
            chrome_options = chromeOption()
            # chrome_options.add_argument('user-data-dir={profile_dir}'.format(profile_dir="chrome_data/"))
            chrome_options.add_argument(
                "User-Agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
            )
            # self.chrome_options.add_argument("--disable-extensions")
            # self.chrome_options.add_argument("--disable-gpu")
            # self.chrome_options.add_argument("--no-sandbox") # linux only
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            # chrome_options.add_argument("--log-path=/home/hanuvadiya/TLP/tlp-scripts/chrome.log")
            chrome_options.add_argument("--headless")
            chrome_options.add_experimental_option(
                "excludeSwitches", ["enable-automation"]
            )
            chrome_options.add_experimental_option("useAutomationExtension", False)
            print("chrome driver path >> ", chrome_driver_path)
            # driver = webdriver.Remote("http://localhost:4444", options=chrome_options,
            driver = webdriver.Remote(
                docker_url,
                options=chrome_options,
                #                                      service_args=[
                #                                          '--verbose',
                #                                          '--log-path=' + chrome_log
                #                                      ]
            )

        driver.implicitly_wait(5)
        print("Check 1-1")
        driver.maximize_window()
        driver.execute_script("document.body.style.zoom='50%'")
        driver.set_window_size(1920, 1080)
        size = driver.get_window_size()
        print(
            "Window size: width = {}px, height = {}px".format(
                size["width"], size["height"]
            )
        )
        return driver
