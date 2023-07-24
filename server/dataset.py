import sys
sys.path.append('..')

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from server.database import Base



class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)

    def __repr__(self):
        return f"<Dataset {self.id}: {self.name}>"