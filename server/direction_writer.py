from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from database import SessionLocal
from direction import add_direction
from model import Model
from resid import Resid
from resid_writer import model_name, cache, cfg
from scaler import add_scaler
from utils import get_layer_num_from_resid_type
import numpy as np


def main():
    sess = SessionLocal()
    model = sess.query(Model).filter(Model.name == model_name).one_or_none()

    for key in cache.keys():
        layer_num = get_layer_num_from_resid_type(key)

        for head in [None, *range(cfg.n_heads)]:
            resids = (
                sess.query(Resid)
                .filter(Resid.model == model)
                .filter(Resid.layer == layer_num)
                .filter(Resid.type == key)
                .filter(Resid.head == head)
                .filter(Resid.token_position > 0)  # The leading |<endoftext>| token is weird
                .all()
            )

            if len(resids) == 0:
                continue

            print(key, head)


            X = np.array([resid.resid for resid in resids])

            print(X.shape)

            standard_scaler = StandardScaler()
            X_scaled = standard_scaler.fit_transform(X)

            # Perform PCA
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)

            explained_variance_ratio = pca.explained_variance_ratio_

            scaler = add_scaler(
                sess,
                standard_scaler=standard_scaler,
                model=model,
                layer=layer_num,
                type=key,
                head=head,
                no_commit=True
            )

            for component_index, component in enumerate(pca.components_):
                add_direction(
                    sess,
                    direction=component,
                    model=model,
                    layer=layer_num,
                    type=key,
                    head=head,
                    generated_by_process='pca',
                    component_index=component_index,
                    scaler=scaler,
                    fraction_of_variance_explained=explained_variance_ratio[component_index],
                    no_commit=True
                )
            
            sess.commit()


if __name__ == '__main__':
    main()