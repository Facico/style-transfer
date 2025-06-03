import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def Headline(sent: str, note_token='*'):
    res = note_token * 20 + sent + note_token * 20
    return res


def visualize(c:str='TSNE', type:str='ori_', alpha: float=0.04):
    h_data = np.load(type + 'h_data.npy')
    z_data = np.load(type + 'z_data.npy')
    label_data = np.load(type + 'label_data.npy')

    print(f'Read {len(h_data)} sentences')

    target = 3
    for target in range(50, 100):
        parallel_distance = ((h_data[target] - h_data[target + 1000]) ** 2).mean()
        non_distance_list = []
        for i in range(1000):
            non_distance_list.append(((h_data[target] - h_data[1000 + i]) ** 2).mean())
        print('pa:', parallel_distance, ' no:', np.array(non_distance_list).mean(), sep=' ')

    pair = [[i] for i in range(20, 27)]
    pair = [w + [w[0] + 1000] for w in pair]
    print(pair)

    if c == 'TSNE':
        perplexity = 60
        h_data = TSNE(n_components=2, perplexity=perplexity).fit_transform(h_data)
        z_data = TSNE(n_components=2, perplexity=perplexity).fit_transform(z_data)
    elif c == 'PCA':
        h_data = PCA(n_components=2).fit_transform(h_data)
        z_data = PCA(n_components=2).fit_transform(z_data)
    else:
        raise ValueError('Dimension Lowering Method not recognized')


    fig, ax = plt.subplots(nrows=3, ncols=2)
    for cur_ax in ax.flatten():
        cur_ax.xaxis.set_major_formatter(NullFormatter())
        cur_ax.yaxis.set_major_formatter(NullFormatter())
    ax[0,0].scatter(h_data[label_data == 0][:,0], h_data[label_data == 0][:,1], c='r', label='0:Negative', alpha=alpha)
    ax[0,0].scatter(h_data[label_data == 1][:,0], h_data[label_data == 1][:,1], c='b', label='1:Positive', alpha=alpha)

    for it, p in enumerate(pair):
        ax[0,0].scatter(h_data[p][:,0], h_data[p][:,1], label=f'Pair{it}', s=70)
    ax[0,0].set_title('Hidden Space')

    ax[0,1].scatter(z_data[label_data == 0][:,0], z_data[label_data == 0][:,1], c='r', label='0:Negative')
    ax[0,1].scatter(z_data[label_data == 1][:,0], z_data[label_data == 1][:,1], c='b', label='1:Positive')
    ax[0,1].set_title('Z Space')

    ax[1,0].scatter(h_data[label_data == 0][:,0], h_data[label_data == 0][:,1], c='r')
    ax[1,0].set_xlim(ax[0,0].get_xlim())
    ax[1,0].set_ylim(ax[0,0].get_ylim())

    ax[1,1].scatter(z_data[label_data == 0][:,0], z_data[label_data == 0][:,1], c='r')
    ax[1,1].set_xlim(ax[0,1].get_xlim())
    ax[1,1].set_ylim(ax[0,1].get_ylim())

    ax[2,0].scatter(h_data[label_data == 1][:,0], h_data[label_data == 1][:,1], c='b')
    ax[2,0].set_xlim(ax[0,0].get_xlim())
    ax[2,0].set_ylim(ax[0,0].get_ylim())

    ax[2,1].scatter(z_data[label_data == 1][:,0], z_data[label_data == 1][:,1], c='b')
    ax[2,1].set_xlim(ax[0,1].get_xlim())
    ax[2,1].set_ylim(ax[0,1].get_ylim())

    # plt.legend()
    # ax[0,1].legend()
    # ax[0,0].legend()
    plt.savefig(f'./fig/fig_{c}_{type}.png')
    plt.show()


if __name__ == '__main__':
    # visualize()
    alpha = 0.05
    if True:
        alpha = 0

    visualize(c='PCA', type='con_', alpha=alpha)
    # visualize(c='PCA', alpha=alpha)
    # visualize(c='TSNE', type='con_', alpha=alpha)
    # visualize(c='TSNE', alpha=alpha)
    # visualize_vae_TSNE()
    # visualize_PCA()
    # visualize_TSNE()
