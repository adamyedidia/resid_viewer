from sqlalchemy import Column, DateTime, ForeignKey, func, Index, Integer, ARRAY, Float, String
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, index=True, unique=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)

    def __repr__(self):
        return f"<User {self.id}: {self.name}>"
    
    def to_json(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at,
        }
    

def add_or_get_user(sess, username: str, no_commit: bool = False) -> User:
    user = User.query.filter_by(name=username).first()
    if user is None:
        user = User(name=username)
        sess.add(user)

        if not no_commit:
            sess.commit()
    return user