"""Add tables

Revision ID: 9f287d5e4246
Revises: bde5b43db960
Create Date: 2023-07-06 22:16:23.858885

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9f287d5e4246'
down_revision = 'bde5b43db960'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('vectors',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('vector', sa.ARRAY(sa.Float()), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_vectors_id'), 'vectors', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_vectors_id'), table_name='vectors')
    op.drop_table('vectors')
    # ### end Alembic commands ###
