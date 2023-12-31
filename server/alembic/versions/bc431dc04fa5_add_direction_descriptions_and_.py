"""Add direction descriptions and component_index column

Revision ID: bc431dc04fa5
Revises: c91c5c02be59
Create Date: 2023-07-07 19:23:37.636083

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bc431dc04fa5'
down_revision = 'c91c5c02be59'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('direction_descriptions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('direction_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['direction_id'], ['directions.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_direction_descriptions_created_at'), 'direction_descriptions', ['created_at'], unique=False)
    op.create_index(op.f('ix_direction_descriptions_direction_id'), 'direction_descriptions', ['direction_id'], unique=False)
    op.create_index(op.f('ix_direction_descriptions_user_id'), 'direction_descriptions', ['user_id'], unique=False)
    op.add_column('directions', sa.Column('component_index', sa.Integer(), nullable=True))
    op.create_index('idx_component_index_created_at', 'directions', ['component_index', 'created_at'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('idx_component_index_created_at', table_name='directions')
    op.drop_column('directions', 'component_index')
    op.drop_index(op.f('ix_direction_descriptions_user_id'), table_name='direction_descriptions')
    op.drop_index(op.f('ix_direction_descriptions_direction_id'), table_name='direction_descriptions')
    op.drop_index(op.f('ix_direction_descriptions_created_at'), table_name='direction_descriptions')
    op.drop_table('direction_descriptions')
    # ### end Alembic commands ###
