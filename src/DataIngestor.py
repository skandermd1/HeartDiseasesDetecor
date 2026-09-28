import pandas as pd



class DataIngestor():
    def ingest(self,file_path:str)->pd.DataFrame:
        if not file_path.endswith(".csv"):
            raise ValueError('file must be .csv')
        df=pd.read_csv(file_path)
        return df