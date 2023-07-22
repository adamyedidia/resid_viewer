
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from transformer_lens import HookedTransformer

sys.path.append('..')





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

    fig, axs = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={'projection':'3d'})


    model_names = ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl",
                "stanford-crfm/alias-gpt2-small-x21", "stanford-crfm/battlestar-gpt2-small-x49",
                "stanford-crfm/caprica-gpt2-small-x81", "stanford-crfm/darkmatter-gpt2-small-x343",
                "stanford-crfm/expanse-gpt2-small-x777"]

    for ax, model_name in zip(axs.flatten(), model_names):

        reference_gpt2 = HookedTransformer.from_pretrained(model_name, fold_ln=False, center_unembed=False,
                                                center_writing_weights=False)


        matrix = reference_gpt2.W_pos.detach().numpy()[1:100]

        print(matrix.shape)

        # Calculate the mean vector and subtract it from the matrix
        mean_vector = np.mean(matrix, axis=0)
        matrix = matrix - mean_vector

        pca = PCA(n_components=6)
        pca = PCA(n_components=)

        pca_result = pca.fit_transform(matrix)

        # Separating the 3 PCA components
        x_pca = pca_result[:, 3]
        y_pca = pca_result[:, 4]
        z_pca = pca_result[:, 5]

        # create color map
        num_of_rows = matrix.shape[0]
        colors = plt.cm.jet(np.linspace(0,1,num_of_rows))

        ax.view_init(elev=20, azim=45)

        for i in range(num_of_rows):
            ax.scatter(x_pca[i], y_pca[i], z_pca[i], color=colors[i])

        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_zlabel("PCA3")

        # compute the variance explained
        variance_explained = sum(pca.explained_variance_ratio_)
        
        # set the title for the subplot
        ax.set_title(f'{model_name} ({variance_explained*100:.2f}% variance)')

    plt.tight_layout()
    plt.savefig('helix_grid_4_thru_6.png')