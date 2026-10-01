import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
import sys

logger = get_logger("etracting_data")




def readData(fileType = "raw") -> pd.DataFrame: 

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "data" 
    DATA_DIR =  fileType if fileType in ['raw', 'processed'] else 'raw'
    FILE_NAME = "diabetesDataProcessed.csv" if fileType == "processed" else 'diabetesDataset.csv' if fileType=='raw' else 'cleanedDataset.csv'

    filePath = DATA_PATH / DATA_DIR / FILE_NAME

    if not filePath.exists(): 
        logger.critical(f"source file is missing")
        sys.exit(1)

    try: 
        deliveryData = pd.read_csv(filePath, na_values=['nan', 'NaN', 'NAN', 'NA', 'na'], keep_default_na=True)
        return deliveryData
    except Exception as e:
        logger.error(f"error while trying to read data: {e}")
        return