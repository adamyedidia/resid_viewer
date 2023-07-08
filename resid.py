from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, ARRAY, Float, String, func
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
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False, index=True)
    prompt = relationship("Prompt", lazy="joined")

    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)

    __table_args__ = (
        Index("idx_resids_model_prompt_layer_type_created_at", "model_id", "prompt_id", "layer", "type", "created_at"),
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
            'prompt': self.prompt.text,
        }