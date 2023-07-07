from IPython import embed
from database import SessionLocal, engine
from vector import Vector

def main():
    sess = SessionLocal()

    # add objects that you want to use in the shell to this dictionary
    user_ns = {"sess": sess, "Vector": Vector}

    print("Starting IPython shell with session and Vector imported.")
    embed(user_ns=user_ns)

    sess.close()

if __name__ == "__main__":
    main()
