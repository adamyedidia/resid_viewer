from sqlalchemy import ARRAY, Column, DateTime, Index, Integer, String, func
from sqlalchemy.orm import relationship
from database import Base

class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True)
    text = Column(String, nullable=False, index=True)

    encoded_text_split_by_token = Column(ARRAY(Integer), nullable=False)
    text_split_by_token = Column(ARRAY(String), nullable=False)

    length_in_tokens = Column(Integer, nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)

    __table_args__ = (
        Index("idx_length_in_tokens_created_at", "length_in_tokens", "created_at"),
    )

    def __repr__(self):
        abbreviated_prompt = self.text[:20] + "..." if len(self.text) > 20 else self.prompt  # type: ignore
        return f"<Prompt {self.id}: {abbreviated_prompt}>"
    
    def to_json(self):
        return {
            'id': self.id,
            'text': self.text,
            'created_at': self.created_at,
        }