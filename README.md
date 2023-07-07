This project runs on Python 3.11.

To install requirements, run

`pip install -r requirements.txt`

Then install postgres:

`brew install postgresql`

Create a database and a corresponding user. From the terminal, run,

`createdb rv`

`psql rv`

From within the PSQL shell that you just opened, run:

`CREATE ROLE rv WITH LOGIN PASSWORD 'resids' CREATEDB CREATEROLE;`

Do CTRL-D to leave the shell.

Then run:

`alembic upgrade head`

to update your local database.

Also, you'll need to install alembic:

`pip install alembic`