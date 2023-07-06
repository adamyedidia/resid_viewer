from sqlalchemy import Column, Integer, ARRAY, Float
from database import Base

class Vector(Base):
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True, index=True)
    vector = Column(ARRAY(Float))