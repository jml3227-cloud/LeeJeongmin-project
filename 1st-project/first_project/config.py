import os
import oracledb

oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")

BASE_DIR = os.path.dirname(__file__)

SECRET_KEY = 'dev'

USER = "scott"
PASSWORD = "tiger"
DSN = "localhost:1521/xe"

MODEL_PATH = os.path.join(BASE_DIR, "random_forest.pkl")
EXPLAINER_PATH = os.path.join(BASE_DIR, "shap_explainer.pkl")