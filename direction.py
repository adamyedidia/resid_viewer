from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, ARRAY, Float, String, func
from database import Base
from sqlalchemy.orm import relationship
import numpy as np
from user import User

class Direction(Base):
    __tablename__ = "directions"

    id = Column(Integer, primary_key=True)
    direction = Column(ARRAY(Float))

    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    model = relationship("Model", lazy="joined")

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    user = relationship("User")

    layer = Column(Integer)
    type = Column(String)

    generated_by_process = Column(String)
    component_index = Column(Integer)

    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)

    __table_args__ = (
        Index("idx_directions_model_layer_type_created_at", "model_id", "layer", "type", "created_at"),
        Index("idx_generated_by_process_created_at", "generated_by_process", "created_at"),
        Index("idx_component_index_created_at", "component_index", "created_at"),
    )

    @property
    def arr(self):
        return np.array(self.direction)

    def __repr__(self):
        return f"<Direction {self.id}: {np.array(self.direction).shape}>"
    
    def to_json(self):
        return {
            'id': self.id,
            'direction': self.direction,
            'model': self.model.name,
            'layer': self.layer,
            'type': self.type,
        }