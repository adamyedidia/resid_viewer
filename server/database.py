import sys
sys.path.append('..')

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from server.settings import DATABASE_URL



engine = create_engine(DATABASE_URL, pool_size=100, max_overflow=100)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()