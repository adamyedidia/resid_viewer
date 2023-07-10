from typing import Optional
from sklearn.discriminant_analysis import StandardScaler
from sqlalchemy import Column, DateTime, ForeignKey, and_, exists, func, Index, Integer, ARRAY, Float, String
from database import Base
from sqlalchemy.orm import relationship
import numpy as np

from model import Model


# Not a misspelling: this is Scaler as in StandardScaler, not Scalar as in a single value
class Scaler(Base):
    __tablename__ = "scalers"

    id = Column(Integer, primary_key=True)

    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)
    model = relationship("Model")

    layer = Column(Integer)
    type = Column(String, nullable=False)
    head = Column(Integer)

    mean = Column(ARRAY(Float), nullable=False)
    scale = Column(ARRAY(Float), nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)

    __table_args__ = (
        Index("idx_scalers_model_layer_type_created_at", "model_id", "layer", "type", "head", "created_at"),
    )

    def __repr__(self):
        return f"<Scaler {self.id}: {self.model.name} {self.layer} {self.type} {self.head}>"


def add_scaler(sess,
               standard_scaler: StandardScaler,
               model: Model,
               layer: Optional[int],
               type: str,
               head: Optional[int],
               no_commit: bool = False) -> Scaler:
    
    if (scaler := sess.query(Scaler).filter(and_(
        Scaler.model == model,
        Scaler.layer == layer,
        Scaler.type == type,
        Scaler.head == head,        
    )).one_or_none()) is not None:
        print('Scaler already exists')
        return scaler

    scaler = Scaler(
        model=model,
        layer=layer,
        type=type,
        head=head,
        mean=standard_scaler.mean_,
        scale=standard_scaler.scale_,
    )

    if not no_commit:
        sess.commit()
    
    sess.add(scaler)
    sess.commit()

    return scaler
    

def get_scaler_object(scaler: Scaler) -> StandardScaler:
    standard_scaler = StandardScaler()
    standard_scaler.mean_ = np.array(scaler.mean)
    standard_scaler.scale_ = np.array(scaler.scale)
    return standard_scaler
