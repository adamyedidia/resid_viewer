"""Add scalers table

Revision ID: 1af84702a4ce
Revises: 4dca427ac24b
Create Date: 2023-07-10 15:25:40.395440

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1af84702a4ce'
down_revision = '4dca427ac24b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('scalers',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('model_id', sa.Integer(), nullable=False),
    sa.Column('layer', sa.Integer(), nullable=True),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('head', sa.Integer(), nullable=True),
    sa.Column('dimension', sa.Integer(), nullable=False),
    sa.Column('mean', sa.ARRAY(sa.Float()), nullable=False),
    sa.Column('scale', sa.ARRAY(sa.Float()), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_scalers_model_layer_type_created_at', 'scalers', ['model_id', 'layer', 'type', 'head', 'created_at'], unique=False)
    op.create_index(op.f('ix_scalers_created_at'), 'scalers', ['created_at'], unique=False)
    op.create_index(op.f('ix_scalers_model_id'), 'scalers', ['model_id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_scalers_model_id'), table_name='scalers')
    op.drop_index(op.f('ix_scalers_created_at'), table_name='scalers')
    op.drop_index('idx_scalers_model_layer_type_created_at', table_name='scalers')
    op.drop_table('scalers')
    # ### end Alembic commands ###
