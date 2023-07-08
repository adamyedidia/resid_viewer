"""Add added_by_user column to prompts

Revision ID: fdf7656a3747
Revises: 4cacc64be527
Create Date: 2023-07-08 14:45:31.261571

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fdf7656a3747'
down_revision = '4cacc64be527'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('prompts', sa.Column('added_by_user_id', sa.Integer(), nullable=True))
    op.create_index('idx_dataset_added_by_user_created_at', 'prompts', ['added_by_user_id', 'created_at'], unique=False)
    op.create_foreign_key('prompts_added_by_user_id_fkey', 'prompts', 'users', ['added_by_user_id'], ['id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('prompts_added_by_user_id_fkey', 'prompts', type_='foreignkey')
    op.drop_index('idx_dataset_added_by_user_created_at', table_name='prompts')
    op.drop_column('prompts', 'added_by_user_id')
    # ### end Alembic commands ###