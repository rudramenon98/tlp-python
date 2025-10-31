import urllib.parse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class MySQLDriver:
    def __init__(self, cred):
        conn_str = (
            "mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
        )
        conn_str = conn_str.format(
            username=urllib.parse.quote(cred["username"]),
            password=urllib.parse.quote(cred["password"]),
            host=urllib.parse.quote(cred["host"]),
            port=cred["port"],
            database=cred["database"],
        )
        print(conn_str)
        self.engine = create_engine(
            conn_str,
            echo=False,
            connect_args={"connect_timeout": 120},
            pool_pre_ping=True,
            pool_size=10,  # default is 5
            max_overflow=20,
            pool_timeout=30,
        )
        self.Session = sessionmaker()

    def get_session(self):
        return self.Session(bind=self.engine)

    def get_connection(self):
        connection = self.engine.connect()
        return connection

    def close(self):
        self.engine.dispose()
