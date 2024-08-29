from matplotlib import pyplot as plt


def deorder_total_dist(total_dist):
    total_dist_non_ordered = dict()
    for item in total_dist:
        key = tuple(sorted(item))
        if key not in total_dist_non_ordered:
            total_dist_non_ordered[key] = 0
        total_dist_non_ordered[key] += total_dist[item]
    return total_dist_non_ordered


def plot_dist_evolution(plot_dist,
                        x=None,
                        mapping=None,
                        save_dir=None,
                        show=False,
                        title=None,
                        xlabel='Epochs',
                        legend_position=(-.3, .5),
                        info=None,
                        info_position=(.0, 1.05),
                        fontsize=None,
                        **kwargs
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
    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots()
    if x is None:
        x = range(len(plot_dist))
    if mapping is not None:
        plot_dist = [mapping(dist) for dist in plot_dist]

    num_pops = len(plot_dist[0])
    for i in range(num_pops):
        kw = {key: kwargs[key][i] for key in kwargs
              if (i < len(kwargs[key]) and kwargs[key][i] is not None)}
        plt.fill_between(x=x,
                         y1=[sum(dist[:i]) for dist in plot_dist],
                         y2=[sum(dist[:i + 1]) for dist in plot_dist],
                         **kw,
                         )
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.legend(loc='center left', bbox_to_anchor=legend_position)
    if info is not None:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        x, y = info_position
        plt.text(x, y, info, verticalalignment='bottom', bbox=props, transform=ax.transAxes)
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
