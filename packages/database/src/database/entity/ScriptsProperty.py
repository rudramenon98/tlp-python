import json
import re
import traceback


def extract_database_name(jdbc_url: str) -> str:
    # Decode any unicode escape sequences like \u003d and \u0026
    decoded_url = jdbc_url.encode("utf-8").decode("unicode_escape")

    # Regex to extract database name (after the host:port/ and before ?)
    match = re.search(r"jdbc:mysql://[^/]+/([^?]+)", decoded_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Database name not found in the JDBC URL.")


def parse_jdbc_mysql_url(jdbc_url):
    # Regex to match hostname, port, and database name in the JDBC MySQL URL
    pattern = r"jdbc:mysql://([^:/?#]+):(\d+)/([^?]+)"
    match = re.search(pattern, jdbc_url)
    if match:
        hostname = match.group(1)
        port = int(match.group(2))
        database = match.group(3)
        return hostname, port, database
    else:
        raise ValueError("Invalid JDBC MySQL URL format")


class DatabaseConfig:
    host: str
    username: str
    password: str
    port: int
    database: str

    def parseDictORIG(self, dictObj):
        self.host = dictObj["dbHost"]
        self.port = int(dictObj["port"])
        self.database = extract_database_name(self.host)
        self.username = dictObj["dbUser"]
        self.password = dictObj["dbPassword"]

    def parseDict(self, dictObj):
        self.host, self.port, self.database = parse_jdbc_mysql_url(dictObj["dbHost"])
        self.username = dictObj["dbUser"]
        self.password = dictObj["dbPassword"]


class ScriptsConfig:
    def __init__(self):
        self.baseDir = None

        self.dockerPrefix = None

        self.databaseConfig = None

        # self.rootDataDir = None
        self.downloadDir = None

        self.pythonDir = None
        self.SeleniumDocker = None
        self.ParserDocker = None
        self.EncoderDocker = None

        self.IndexDir = None
        self.DictionaryDocker = None
        self.StaticIndexerDocker = None
        self.DynamicIndexerDocker = None
        self.keywordServerDocker = None
        self.LogstashDocker = None
        self.GenAIDocker = None
        self.ServerHost = None

    def parseDict(self, dictObj):
        try:
            #            print(dictObj)
            #            print(type(dictObj))
            if isinstance(dictObj, str):
                dictObj = json.loads(dictObj)

            #            print(dictObj['loginEntity']['baseFolderPath'])
            self.baseDir = dictObj["loginEntity"]["baseFolderPath"]
            # self.dockerPrefix = dictObj['docker-prefix']
            self.databaseConfig = DatabaseConfig()
            self.databaseConfig.parseDict(dictObj=dictObj["dbConfigurations"])

            try:
                self.ServerHost = dictObj["SERVER_HOST"]
            except KeyError:
                pass

            self.StaticIndexerDocker = dictObj["dockerSettings"][
                "staticIndexServerAddress"
            ]
            self.DynamicIndexerDocker = dictObj["dockerSettings"][
                "dynamicIndexServerAddress"
            ]
            self.keywordServerDocker = dictObj["dockerSettings"]["keywordServerAddress"]
            # self.logstashDocker = dictObj['dockerSettings']['logStashServerAddress']
            self.SeleniumDocker = dictObj["dockerSettings"][
                "seleniumChromeServerAddress"
            ].rstrip("/")
            self.ParserDocker = dictObj["dockerSettings"]["parserServerAddress"]
            self.EncoderDocker = dictObj["dockerSettings"]["encoderOllamaServerAddress"]
            self.GenAIDocker = dictObj["dockerSettings"]["genAIOllamaServerAddress"]
            self.DictionaryDocker = dictObj["dockerSettings"]["dictionaryServerAddress"]

            # if self.scannDocker:
            #    self.dockerPrefix = self.scannDocker.lower().split('_')[0].lower()

            if self.baseDir is not None:
                # self.parserModel = self.etl_baseDir + '/models/AIPDFParserModel/RandomForestClassifier_modeltest_Production_0517920231.sav'
                self.parserModel = (
                    self.baseDir
                    + "/models/AIPDFParserModel/RandomForestClassifier_modeltest_Production_082920231.1.sav"
                )
                self.BertModelDir = self.baseDir + "/languageModel/"

                self.downloadDir = self.baseDir + "/scrapedDocs/"
                self.pythonDir = self.baseDir + "/../scripts/"
                self.IndexDir = self.baseDir + "/index/"
                self.S2VModelDir = self.baseDir + "/models/S2V/"
                self.spacyModelDir = self.baseDir + "/models/NER/"

        except Exception:
            print("ERROR: loading Command line parameters")
            traceback.print_exc()


def parseCredentialFile(credFile):
    try:
        with open(credFile, "r") as f:
            file_text = f.read()
            props = json.loads(file_text)
            configs = ScriptsConfig()
            configs.parseDict(props)
            return configs
    except Exception:
        print("ERROR: loading configuration parameters")
        traceback.print_exc()
        return None
