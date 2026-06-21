from sqlalchemy import create_engine


class DatabaseManager:
    def __init__(self, database_path="output/results.db"):
        self.database_path = database_path
        self.engine = create_engine(f"sqlite:///{self.database_path}")

    def save_dataframe(self, dataframe, table_name):
        dataframe.to_sql(
            table_name,
            self.engine,
            if_exists="replace",
            index=False
        )