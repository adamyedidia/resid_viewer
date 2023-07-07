from sqlalchemy import Column, DateTime, ForeignKey, func, Index, Integer, ARRAY, Float, String
from database import Base
from sqlalchemy.orm import relationship

class DirectionDescription(Base):
    __tablename__ = "direction_descriptions"

    id = Column(Integer, primary_key=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", lazy="joined")

    direction_id = Column(Integer, ForeignKey("directions.id"), nullable=False, index=True)
    direction = relationship("Direction", lazy="joined")

    def __repr__(self):
        return f"<DirectionDescription {self.id} by {self.user.name} of Direction {self.direction.id}>"
    
    def to_json(self):
        return {
            'id': self.id,
            'created_at': self.created_at,
            'user': self.user.name,
            'direction_id': self.direction.id,
        }
    