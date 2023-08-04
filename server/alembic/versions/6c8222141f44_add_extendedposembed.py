"""Add ExtendedPosEmbed

Revision ID: 6c8222141f44
Revises: 08ca044a6049
Create Date: 2023-08-03 18:13:35.729217

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6c8222141f44'
down_revision = '08ca044a6049'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('extended_pos_embeds',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('extended_pos_embed', sa.ARRAY(sa.Float()), nullable=True),
    sa.Column('model_id', sa.Integer(), nullable=False),
    sa.Column('layer', sa.Integer(), nullable=True),
    sa.Column('token_position', sa.Integer(), nullable=True),
    sa.Column('tokens_used_to_compute', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('tokens_used_to_compute_hash', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_model_layer_hash_token_position', 'extended_pos_embeds', ['model_id', 'layer', 'tokens_used_to_compute_hash', 'token_position'], unique=False)
    op.create_index(op.f('ix_extended_pos_embeds_created_at'), 'extended_pos_embeds', ['created_at'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_extended_pos_embeds_created_at'), table_name='extended_pos_embeds')
    op.drop_index('idx_model_layer_hash_token_position', table_name='extended_pos_embeds')
    op.drop_table('extended_pos_embeds')
    # ### end Alembic commands ###
