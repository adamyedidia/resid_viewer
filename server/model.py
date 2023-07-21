from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from server.database import Base

class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)

    def __repr__(self):
        return f"<Model {self.id}: {self.name}>"