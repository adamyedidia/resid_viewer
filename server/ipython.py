import sys
sys.path.append('..')

from IPython import embed
from server.database import SessionLocal, engine
from server.resid import Resid
from server.model import Model
from server.user import User
from server.direction import Direction
from server.direction_description import DirectionDescription
from server.prompt import Prompt
from sqlalchemy import func
from server.utils import enc

def main():
    sess = SessionLocal()

    gpt2_small = sess.query(Model).filter(Model.name == "gpt2-small").one_or_none()

    # add objects that you want to use in the shell to this dictionary
    user_ns = {
        "sess": sess, 
        "Resid": Resid,
        "Model": Model,
        "User": User,
        "Direction": Direction,
        "DirectionDescription": DirectionDescription,
        "Prompt": Prompt,
        "func": func,
        "gpt2_small": gpt2_small,
        "enc": enc,
    }

    embed(user_ns=user_ns)

    sess.close()

if __name__ == "__main__":
    main()
