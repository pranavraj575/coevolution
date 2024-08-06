from matplotlib import pyplot as plt


def plot_dist_evolution(plot_dist,
                        x=None,
                        mapping=None,
                        save_dir=None,
                        show=False,
                        alphas=None,
                        colors=None,
                        labels=None,
                        title=None,
                        info=None,
                        ):
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
    fig, ax = plt.subplots()
    if x is None:
        x = range(len(plot_dist))
    if mapping is not None:
        plot_dist = [mapping(dist) for dist in plot_dist]

    num_pops = len(plot_dist[0])
    kwargs = dict()
    if alphas is not None:
        kwargs['alpha'] = alphas
    if colors is not None:
        kwargs['color'] = colors
    if labels is not None:
        kwargs['label'] = labels
    for i in range(num_pops):
        plt.fill_between(x=x,
                         y1=[sum(dist[:i]) for dist in plot_dist],
                         y2=[sum(dist[:i + 1]) for dist in plot_dist],
                         **{key: kwargs[key][i] for key in kwargs}
                         )
    if title is not None:
        plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(-.3, .5))
    if info is not None:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.text(.0, 1.05, info, verticalalignment='bottom', bbox=props, transform=ax.transAxes)
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
