import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
import sys

logger = get_logger("extracting_data")




def readData(data = "raw") -> pd.DataFrame: 

    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "data" 

    paths = {
        'raw' : 'raw/raw_data.csv',
        'clean' : 'clean/clean_data.csv',
        'processed' : 'processed/processed_data.csv',
        'cluster' : 'processed/cluster_data.csv'
    }

    filePath = DATA_PATH / paths[data]

    print(filePath)

    if not filePath.exists(): 
        logger.critical(f"source file is missing")
        return

    try: 
        deliveryData = pd.read_csv(filePath, na_values=['nan', 'NaN', 'NAN', 'NA', 'na'], keep_default_na=True)
        return deliveryData
    except Exception as e:
        logger.error(f"error while trying to read data: {e}")
        return