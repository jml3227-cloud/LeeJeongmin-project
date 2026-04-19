import os
import oracledb
from dotenv import load_dotenv

load_dotenv()

oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")

BASE_DIR = os.path.dirname(__file__)

SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
USER = os.environ.get('DB_USER')
PASSWORD = os.environ.get('DB_PASSWORD')
DSN = os.environ.get('DB_DSN')

MODEL_PATH = os.path.join(BASE_DIR, "random_forest.pkl")
EXPLAINER_PATH = os.path.join(BASE_DIR, "shap_explainer.pkl")