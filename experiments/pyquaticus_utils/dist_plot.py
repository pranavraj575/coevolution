from matplotlib import pyplot as plt
import numpy as np
import torch, copy


def order_compensate_dist(dist):
    """
    compensates for ordering issues arising from repeat members
    Args:
        dist: dict(tuple -> probability)
    Returns:
        copy of dist, compensated for order then normalized
    """
    dist = copy.deepcopy(dist)
    some = 0
    for team in dist:
        compensation = torch.tensor(1.)
        for member in team:
            # compared to a string of all unique elements, this string is undercounted by a factor of
            #  prod(n_i!) for n_1 repeats of the first unique member, n_2 of the second, ...
            # to account for this, we multiply by this number
            # we take the root so we dont have to do logic like only consider the first occurrence of each
            cnt = team.count(member)
            compensation = compensation*torch.pow(torch.prod(torch.arange(cnt) + 1), 1/cnt)
        dist[team] = compensation.item()*dist[team]
        some += dist[team]
    for team in dist:
        dist[team] = dist[team]/some
    return dist


def deorder_total_dist(total_dist):
    total_dist_non_ordered = dict()
    for item in total_dist:
        key = tuple(sorted(item))
        if key not in total_dist_non_ordered:
            total_dist_non_ordered[key] = 0
        total_dist_non_ordered[key] += total_dist[item]
    return total_dist_non_ordered


def smooth_backwards(data, r=1):
    out = data.copy()
    for i in range(1, r + 1):
        out[:-i] += data[i:]
        out[-i:] += data[-i:]
    out = out/(r + 1)
    return out


def plot_dist_evolution(plot_dist,
                        x=None,
                        mapping=None,
                        save_dir=None,
                        show=False,
                        title=None,
                        inverted=False,
                        xlabel='Epochs',
                        legend_position=(-.3, .5),
                        info=None,
                        info_position=(.0, 1.05),
                        fontsize=None,
                        smoothing=0,
                        **kwargs
                        ):
    """
    Args:
        plot_dist:
        mapping: maps a dist from plot_dist into another dist
            i.e. sum a few of them
        save_dir:
        inverted: True if we put the first element on the top
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

    # ith element is the probability of the ith distribution plus all previous
    cum_dists = [np.array([sum(dist[:i]) for dist in plot_dist])
                 for i in range(1, num_pops + 1)]
    if smoothing > 0:
        cum_dists = [smooth_backwards(d, r=smoothing) for d in cum_dists]

    maxdist = min(len(plot_dist), len(x))
    cum_dists = [d[:maxdist] for d in cum_dists]
    x = x[:maxdist]

    # we go through in inverted order to ensure legend looks nice
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if inverted:
        order = ((enumerate(cum_dists)))
    else:
        order = reversed(list(enumerate(cum_dists)))
    for i, cum_dist in order:
        if i == 0:
            prevdist = np.zeros(maxdist)
        else:
            prevdist = cum_dists[i - 1]
        kw = {key: kwargs[key][i] for key in kwargs
              if (i < len(kwargs[key]) and kwargs[key][i] is not None)}

        if 'color' not in kw and i < len(default_colors):
            kw['color'] = default_colors[i]
        if inverted:
            prevdist, cum_dist = 1 - prevdist, 1 - cum_dist
        plt.fill_between(x=x,
                         y1=prevdist,
                         y2=cum_dist,
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
