from typing import Optional
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, ARRAY, Float, String, and_, exists, func
from sqlalchemy.orm import relationship
from database import Base, SessionLocal
import numpy as np
from sklearn.discriminant_analysis import StandardScaler

from model import Model
from prompt import Prompt

class Resid(Base):
    __tablename__ = "resids"

    id = Column(Integer, primary_key=True, index=True)
    resid = Column(ARRAY(Float), nullable=False)

    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    model = relationship("Model", lazy="joined")

    layer = Column(Integer)
    type = Column(String, nullable=False)
    head = Column(Integer)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    prompt = relationship("Prompt", lazy="joined")

    token_position = Column(Integer, nullable=False)

    dimension = Column(Integer, nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_resids_model_prompt_layer_type_head_token_position_ca", 
              "model_id", "prompt_id", "layer", "type", "head", "token_position", "created_at"),
        Index("idx_resids_model_layer_type_head_token_position_ca",
              "model_id", "layer", "type", "head", "token_position", "created_at"),
    )

    @property
    def encoded_token(self) -> int:
        # The minus one is there because the token position is 1-indexed
        # The 0 position is reserved for the |<endoftext>| token that gets prepended
        # to the prompt
        return self.prompt.encoded_text_split_by_token[self.token_position - 1]
    
    @property
    def decoded_token(self) -> str:
        # The minus one is there because the token position is 1-indexed
        # The 0 position is reserved for the |<endoftext>| token that gets prepended
        # to the prompt
        return self.prompt.text_split_by_token[self.token_position - 1]

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
    

def add_resid(sess,
              arr: np.ndarray, 
              model: Model, 
              prompt: Prompt, 
              layer: Optional[int], 
              type: str,
              token_position: int,
              head: Optional[int] = None,
              no_commit: bool = False) -> None:
    
    assert len(arr.shape) == 1, "Resid must be 1-dimensional"

    if (sess.query(exists().where(and_(
        Resid.model == model,
        Resid.prompt == prompt,
        Resid.layer == layer,
        Resid.type == type,
        Resid.token_position == token_position,
        Resid.head == head,
    ))).scalar()):
        print(f"Resid already exists")
        return

    list_array = [float(i) for i in arr]

    resid = Resid(
        resid=list_array,
        model=model,
        prompt=prompt,
        layer=layer,
        type=type,
        token_position=token_position,
        head=head,
        dimension=arr.shape[0],
    )

    sess.add(resid)
    if not no_commit:
        sess.commit()
