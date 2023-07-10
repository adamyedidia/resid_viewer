from typing import Optional
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, ARRAY, Float, String, and_, exists, func
from database import Base
from sqlalchemy.orm import relationship
import numpy as np
from model import Model
from scaler import Scaler
from user import User

class Direction(Base):
    __tablename__ = "directions"

    id = Column(Integer, primary_key=True)
    direction = Column(ARRAY(Float))

    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    model = relationship("Model", lazy="joined")

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    user = relationship("User")

    name = Column(String, nullable=True, index=True)

    layer = Column(Integer)
    type = Column(String)
    head = Column(Integer)

    generated_by_process = Column(String)
    component_index = Column(Integer)

    scaler_id = Column(Integer, ForeignKey("scalers.id"), nullable=True)
    scaler = relationship("Scaler")

    fraction_of_variance_explained = Column(Float)

    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)

    __table_args__ = (
        Index("idx_generated_by_process_created_at", "generated_by_process", "created_at"),
        Index("idx_directions_model_layer_type_created_at", "model_id", "layer", "type", "head", "component_index", "created_at"),
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
    

def add_direction(sess,
                  direction: np.ndarray,
                  model: Model,
                  layer: Optional[int],
                  type: str,
                  head: int,
                  generated_by_process: str,
                  component_index: Optional[int],
                  scaler: Optional[Scaler],
                  fraction_of_variance_explained: Optional[float],
                  user: Optional[User] = None,
                  name: Optional[str] = None,
                  no_commit: bool = False) -> Direction:
    if user is None:
        if (direction := sess.query(Direction).filter(and_(
            Direction.model == model,
            Direction.layer == layer,
            Direction.type == type,
            Direction.head == head,
            Direction.component_index == component_index,
            Direction.name == name,
        )).one_or_none()) is not None:
            print('Direction already exists')
            return direction  # type: ignore
    
    direction = Direction(
        direction=direction,
        model=model,
        layer=layer,
        type=type,
        head=head,
        generated_by_process=generated_by_process,
        component_index=component_index,
        scaler=scaler,
        fraction_of_variance_explained=fraction_of_variance_explained,
        user=user,
        name=name,
    )
    
    sess.add(direction)
    if not no_commit:
        sess.commit()

    return direction
