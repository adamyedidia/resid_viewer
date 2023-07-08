"""Add some columns to prompts

Revision ID: af74007eb366
Revises: cefbc3c6ce4d
Create Date: 2023-07-08 14:01:18.989031

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'af74007eb366'
down_revision = 'cefbc3c6ce4d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('prompts', sa.Column('encoded_text_split_by_token', sa.ARRAY(sa.Integer()), nullable=False))
    op.add_column('prompts', sa.Column('text_split_by_token', sa.ARRAY(sa.String()), nullable=False))
    op.add_column('prompts', sa.Column('length_in_tokens', sa.Integer(), nullable=False))
    op.create_index('idx_length_in_tokens_created_at', 'prompts', ['length_in_tokens', 'created_at'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('idx_length_in_tokens_created_at', table_name='prompts')
    op.drop_column('prompts', 'length_in_tokens')
    op.drop_column('prompts', 'text_split_by_token')
    op.drop_column('prompts', 'encoded_text_split_by_token')
    # ### end Alembic commands ###
