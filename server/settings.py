import os

DATABASE_URL = "postgresql://rv:resids@localhost:5432/rv"
M1_MAC = True
DEMO = True
NULL_DATASET_IS_OPENWEBTEXT = False

if os.path.exists('local_settings.py'):
    from server.local_settings import *