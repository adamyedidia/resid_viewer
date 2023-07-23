
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from transformer_lens import HookedTransformer
import torch

from sklearn.metrics.pairwise import cosine_similarity


sys.path.append('..')

import tiktoken

enc = tiktoken.get_encoding('r50k_base')


M1_MAC = True


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

    fig, axs = plt.subplots(1, 2, figsize=(15, 15), subplot_kw={'projection':'3d'})



    model_names = ["gpt2-small"]

    position = 500

    for ax, model_name in zip(axs.flatten(), model_names):

        reference_gpt2 = HookedTransformer.from_pretrained(model_name, fold_ln=False, center_unembed=False,
                                                center_writing_weights=False)

        # reference_text = "The greatest president of all time was Abraham"
        # reference_text = 'hey, what is that? is that a dog'
        reference_text = ('Request: Please repeat the following string exactly: "hello" '
                          'Reply: "hello". '
                          'Request: "Please repeat the following string exactly: "gorapopefm" '
                          'Reply: "gorapopefm" '
                          'Request: "Please repeat the following string exactly: "adfgpinaie" '
                          'Reply: "adfgpinaie" '
                          'Request: "Please repeat the following string exactly: " poaspdmfpm" '
                          'Reply: " poaspdmfpm" '
                          'Request: "Please repeat the following string exactly: "wplmedpmdp" '
                          'Reply: "wplmedpmdp" '
                          'Request: "Please repeat the following string exactly: "pvofampovm" '
                          'Reply: "pvofampovm" '
                          'Request: "Please repeat the following string exactly: "poemfvpoe" '
                          'Reply: "poemfvpoe" '
                          'Request: "Please repeat the following string exactly: "vfavn" '
                          'Reply: "vfavn" '
                          'Request: "Please repeat the following string exactly: "sopqmx" '
                          'Reply: "sopqmx" '
                          'Request: "Please repeat the following string exactly: "france" '
                          'Reply: "france" '
                          'Request: "Please repeat the following string exactly: "vilion" '
                          'Reply: "pvofampovm" ' 
                          'Request: "Please repeat the following string exactly: " jack" '
                          'Reply: " jack" '
                          'Request: "Please repeat the following string exactly: "aervaxv" '
                          'Reply: "aervaxv" '
                          'Request: "Please repeat the following string exactly: " poem" '
                          'Reply: " poem" '
                          'Request: "Please repeat the following string exactly: " Reddit" '
                          'Reply: " Reddit" '
                          'Request: "Please repeat the following string exactly: "irnnrf" '
                          'Reply: "irnnrf" '
                          'Request: "Please repeat the following string exactly: "wepoc" '
                          'Reply: "wepoc" '
                          'Request: "Please repeat the following string exactly: "propmfpm" '
                          'Reply: "propmfpm" '
                          'Request: "Please repeat the following string exactly: " Germany" '
                          'Reply: " Germany""'
                          'Request: "Please repeat the following string exactly: "rathasoadga" '
                          'Reply: "rathasoadga" '
                          'Request: "Please repeat the following string exactly: "1pdjpm3efe4" '
                          'Reply: "1pdjpm3efe4" '
                          'Request: "Please repeat the following string exactly: "oulosacters" '
                          'Reply: "')
        
        print(len(enc.encode(reference_text)))
        print([(i, enc.decode([tok])) for i, tok in enumerate(enc.encode(reference_text))])

        tokens = reference_gpt2.to_tokens(reference_text)

        def cuda(x):
            return x.to('cpu') if M1_MAC else x.cuda()

        tokens = cuda(tokens)
        logits, cache = reference_gpt2.run_with_cache(tokens)

        print(cache['blocks.0.hook_resid_pre'].shape)

        resid = cache['blocks.0.hook_resid_pre'][0, -100:, :].detach().numpy()
        # resid = cache['pos_embed'][0, :, :].detach().numpy()
        # resid = reference_gpt2.W_pos.detach().numpy()

        last_logits = logits[-1, -1]  # type: ignore
        # Apply softmax to convert the logits to probabilities
        probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()
        
        # Get the indices of the top 10 probabilities
        topk_indices = np.argpartition(probabilities, -10)[-10:]
        # Get the top 10 probabilities
        topk_probabilities = probabilities[topk_indices]
        # Get the top 10 tokens
        topk_tokens = [enc.decode([i]) for i in topk_indices]

        # Print the top 10 tokens and their probabilities
        for token, probability in zip(topk_tokens, topk_probabilities):
            print(f"Token: {token}, Probability: {probability}")



        matrix = reference_gpt2.W_pos.detach().numpy()

        print(matrix.shape)

        # Calculate the mean vector and subtract it from the matrix
        mean_vector = np.mean(matrix, axis=0)
        matrix = matrix - mean_vector
        resid = resid - mean_vector

        pca = PCA(n_components=6)

        pca_result = pca.fit_transform(matrix)
        resid_result = pca.transform(resid)

        print(pca_result.shape)

        # Separating the 3 PCA components
        # x_pca = pca_result[:, 3]
        # y_pca = pca_result[:, 4]
        # z_pca = pca_result[:, 5]

        vec = pca_result[position] - pca_result[position - 1]

        embed = reference_gpt2.W_E.detach().numpy()

        embed = embed - mean_vector

        pca_embed = pca.transform(embed)  # Transform embed into PCA space

        # Calculate cosine similarities
        cosine_similarities = cosine_similarity(pca_embed, vec.reshape(1, -1))

        # Get the indices of most and least similar vectors
        most_similar_index = np.argmax(cosine_similarities)
        least_similar_index = np.argmin(cosine_similarities)

        print(enc.decode([int(most_similar_index)]))
        print(enc.decode([int(least_similar_index)]))

        # Get the most and least similar vectors
        most_similar_vector = pca_embed[most_similar_index]
        least_similar_vector = pca_embed[least_similar_index]

        print(vec, vec.shape)
        print(most_similar_vector, most_similar_vector.shape)
        print(least_similar_vector, least_similar_vector.shape)
        print(most_similar_vector, cosine_similarity(most_similar_vector.reshape(1, -1), vec.reshape(1, -1)))
        print(least_similar_vector, cosine_similarity(least_similar_vector.reshape(1, -1), vec.reshape(1, -1)))

        x_pca = resid_result[:, 0]
        y_pca = resid_result[:, 1]
        z_pca = resid_result[:, 2]

        # create color map
        num_of_rows = resid.shape[0]
        colors = plt.cm.jet(np.linspace(0,1,num_of_rows))

        ax.view_init(elev=20, azim=45)

        for i in range(num_of_rows):
            ax.scatter(x_pca[i], y_pca[i], z_pca[i], 
                    #    color=colors[i] if i != 94 and i != 95 else 'green' if i == 94 else 'blue')
                       color=colors[i])

        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_zlabel("PCA3")

        # compute the variance explained
        variance_explained = sum(pca.explained_variance_ratio_)
        
        # set the title for the subplot
        ax.set_title(f'{model_name} ({variance_explained*100:.2f}% variance)')

        plt.show()