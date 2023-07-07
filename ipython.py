from IPython import embed
from database import SessionLocal, engine
from vector import Vector
from resid import Resid
from model import Model
from user import User
from direction import Direction

def main():
    sess = SessionLocal()

    # add objects that you want to use in the shell to this dictionary
    user_ns = {
        "sess": sess, 
        "Vector": Vector,
        "Resid": Resid,
        "Model": Model,
        "User": User,
        "Direction": Direction,
    }

    print("Starting IPython shell with session and Vector imported.")
    embed(user_ns=user_ns)

    sess.close()

if __name__ == "__main__":
    main()
