"""Add dataset column

Revision ID: 4cacc64be527
Revises: af74007eb366
Create Date: 2023-07-08 14:38:18.880602

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4cacc64be527'
down_revision = 'af74007eb366'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('prompts', sa.Column('dataset', sa.String(), nullable=True))
    op.create_index('idx_dataset_length_in_tokens_created_at', 'prompts', ['dataset', 'length_in_tokens', 'created_at'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('idx_dataset_length_in_tokens_created_at', table_name='prompts')
    op.drop_column('prompts', 'dataset')
    # ### end Alembic commands ###
