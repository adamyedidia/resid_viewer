"""Tweak resids table

Revision ID: 55a4a8f753cc
Revises: fdf7656a3747
Create Date: 2023-07-09 17:45:44.533380

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '55a4a8f753cc'
down_revision = 'fdf7656a3747'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('resids', sa.Column('head', sa.Integer(), nullable=True))
    op.add_column('resids', sa.Column('token_position', sa.Integer(), nullable=False))
    op.add_column('resids', sa.Column('dimension', sa.Integer(), nullable=False))
    op.alter_column('resids', 'layer',
               existing_type=sa.INTEGER(),
               nullable=True)
    op.drop_index('idx_resids_model_prompt_layer_type_created_at', table_name='resids')
    op.drop_index('ix_resids_created_at', table_name='resids')
    op.drop_index('ix_resids_model_id', table_name='resids')
    op.drop_index('ix_resids_prompt_id', table_name='resids')
    op.create_index('idx_resids_model_prompt_layer_type_head_token_position_ca', 'resids', ['model_id', 'prompt_id', 'layer', 'type', 'head', 'token_position', 'created_at'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('idx_resids_model_prompt_layer_type_head_token_position_ca', table_name='resids')
    op.create_index('ix_resids_prompt_id', 'resids', ['prompt_id'], unique=False)
    op.create_index('ix_resids_model_id', 'resids', ['model_id'], unique=False)
    op.create_index('ix_resids_created_at', 'resids', ['created_at'], unique=False)
    op.create_index('idx_resids_model_prompt_layer_type_created_at', 'resids', ['model_id', 'prompt_id', 'layer', 'type', 'created_at'], unique=False)
    op.alter_column('resids', 'layer',
               existing_type=sa.INTEGER(),
               nullable=False)
    op.drop_column('resids', 'dimension')
    op.drop_column('resids', 'token_position')
    op.drop_column('resids', 'head')
    # ### end Alembic commands ###