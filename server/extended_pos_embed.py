import sys
sys.path.append('..')

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, ARRAY, Float, String, and_, exists, func
from server.database import Base
from sqlalchemy.orm import relationship

from server.model import Model

import numpy as np

class ExtendedPosEmbed(Base):
    __tablename__ = "extended_pos_embeds"

    id = Column(Integer, primary_key=True)
    extended_pos_embed = Column(ARRAY(Float))

    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    model = relationship("Model", lazy="joined")

    layer = Column(Integer, nullable=False)

    type = Column(String, nullable=False)

    token_position = Column(Integer, nullable=False)    

    tokens_used_to_compute = Column(ARRAY(String))

    tokens_used_to_compute_hash = Column(String, nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)

    __table_args__ = (
        Index("idx_model_layer_type_hash_token_position", "model_id", "layer", "tokens_used_to_compute_hash", "token_position"),
    )


def get_extended_pos_embed_matrix(sess, model: Model, layer: int, type: str) -> np.ndarray:
    hash = (
        sess.query(ExtendedPosEmbed.tokens_used_to_compute_hash)
        .filter(and_(ExtendedPosEmbed.model_id == model.id, 
                     ExtendedPosEmbed.layer == layer,
                     ExtendedPosEmbed.type == type))
        .order_by(ExtendedPosEmbed.created_at.desc())
        .first()
    )[0]
    
    extended_pos_embeds = (
        sess.query(ExtendedPosEmbed)
        .filter(and_(ExtendedPosEmbed.model_id == model.id, 
                     ExtendedPosEmbed.layer == layer,
                     ExtendedPosEmbed.type == type,
                     ExtendedPosEmbed.tokens_used_to_compute_hash == hash))
        .order_by(ExtendedPosEmbed.token_position.asc())
        .all()
    )
    extended_pos_embed_matrix = np.array([extended_pos_embed.extended_pos_embed for extended_pos_embed in extended_pos_embeds])
    return extended_pos_embed_matrix