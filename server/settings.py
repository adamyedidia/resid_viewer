import os

DATABASE_URL = "postgresql://rv:resids@localhost:5432/rv"
M1_MAC = True
DEMO = True

if os.path.exists('local_settings.py'):
    from server.local_settings import *