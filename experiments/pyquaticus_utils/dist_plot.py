from matplotlib import pyplot as plt


def plot_dist_evolution(plot_dist,
                        x=None,
                        mapping=None,
                        save_dir=None,
                        show=False,
                        alphas=None,
                        colors=None,
                        labels=None,
                        title=None):
    """
    Args:
        plot_dist:
        mapping: maps a dist from plot_dist into another dist
            i.e. sum a few of them
        save_dir:
        show:
        alphas:
        labels: labels for each pop (or for each mapped pop)
        title:
    """
    if x is None:
        x = range(len(plot_dist))
    if mapping is not None:
        plot_dist = [mapping(dist) for dist in plot_dist]

    num_pops = len(plot_dist[0])
    if alphas is None:
        alphas = [1 for _ in range(num_pops)]
    if colors is None:
        colors = [None for _ in range(num_pops)]
    for i in range(num_pops):
        plt.fill_between(x=x,
                         y1=[sum(dist[:i]) for dist in plot_dist],
                         y2=[sum(dist[:i + 1]) for dist in plot_dist],
                         label=labels[i] if labels is not None and len(labels) >= i else None,
                         alpha=alphas[i],
                         color=colors[i],
                         )
    if title is not None:
        plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(-.25, .5))

    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
