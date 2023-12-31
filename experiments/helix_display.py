
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
sys.path.append('..')
from transformer_lens import HookedTransformer
from server.database import SessionLocal

from server.extended_pos_embed import get_extended_pos_embed_matrix
from server.model import Model

from experiments.experimental_transformer import average_rows





if __name__ == '__main__':

# model_name = "gpt2-small"
# model_name = "gpt2-xl"
# model_name = "gpt2-medium"
# model_name = "gpt2-large"
# model_name = "pythia-70m"
#model_name = "stanford-crfm/alias-gpt2-small-x21"
# model_name = "stanford-crfm/battlestar-gpt2-small-x49"
# model_name = "stanford-crfm/caprica-gpt2-small-x81"
# model_name = "stanford-crfm/darkmatter-gpt2-small-x343"
# model_name = "stanford-crfm/expanse-gpt2-small-x777"

    sess = SessionLocal()

    model = sess.query(Model).filter(Model.name == "gpt2-small").first()

    if not model:
        raise Exception('No model found')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    model_name = "gpt2-small"

    reference_gpt2 = HookedTransformer.from_pretrained(model_name, fold_ln=False, center_unembed=False,
                                            center_writing_weights=False)


    matrix = reference_gpt2.W_pos.detach().numpy()

    # Calculate the mean vector and subtract it from the matrix
    mean_vector = np.mean(matrix, axis=0)
    matrix = matrix - mean_vector

    # print(matrix.shape)

    # matrix2 = average_rows(reference_gpt2.W_pos.detach().numpy(), 10)
    matrix2 = get_extended_pos_embed_matrix(sess, model, 2, 'blocks.2.ln1.hook_normalized')

    # Concatenate matrix and matrix2
    concat_matrix = np.concatenate((matrix, matrix2), axis=0)


    pca = PCA(n_components=3)

    pca_result_concat = pca.fit_transform(concat_matrix)
    # pca_result = pca.fit_transform(matrix)
    pca_result = pca.transform(matrix)
    pca_result2 = pca.transform(matrix2)

    # Separating the 3 PCA components
    x_pca = pca_result[:, 0]
    y_pca = pca_result[:, 1]
    z_pca = pca_result[:, 2]

    x_pca2 = pca_result2[:, 0]
    y_pca2 = pca_result2[:, 1]
    z_pca2 = pca_result2[:, 2]


    # create color map
    num_of_rows = matrix.shape[0]
    # colors = plt.cm.jet(np.linspace(0,1,num_of_rows))

    ax.view_init(elev=20, azim=45)

    for i in range(num_of_rows):
        # ax.scatter(x_pca[i], y_pca[i], z_pca[i], color=colors[i])
        ax.scatter(x_pca[i], y_pca[i], z_pca[i], color='blue')
        ax.scatter(x_pca2[i], y_pca2[i], z_pca2[i], color='red')

    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")

    # compute the variance explained
    variance_explained = sum(pca.explained_variance_ratio_)
    
    # set the title for the subplot
    ax.set_title(f'{model_name} ({variance_explained*100:.2f}% variance)')

    plt.show()

    plt.tight_layout()
    plt.savefig('helix_grid_grouped_vs_not.png')