from sqlalchemy import Column, ForeignKey, Index, Integer, ARRAY, Float, String
from sqlalchemy.orm import relationship
from database import Base
import numpy as np

class Resid(Base):
    __tablename__ = "resids"

    id = Column(Integer, primary_key=True, index=True)
    resid = Column(ARRAY(Float), nullable=False)

    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)
    model = relationship("Model", lazy="joined")

    layer = Column(Integer, nullable=False)
    type = Column(String, nullable=False)

    __table_args__ = (
        Index("idx_resids_model_layer_type", "model_id", "layer", "type"),
    )

    @property
    def arr(self):
        return np.array(self.resid)

    def __repr__(self):
        return f"<Resid {self.id}: {np.array(self.resid).shape}>"
    
    def to_json(self):
        return {
            'id': self.id,
            'resid': self.resid,
            'model': self.model.name,
            'layer': self.layer,
            'type': self.type,
        }