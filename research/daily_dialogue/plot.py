import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import OrderedDict
from cycler import cycler
from sklearn.metrics import precision_recall_curve


def wrap_text(string, size_limit=None):
    if not size_limit:
        return string
    tokens = string.split(' ')
    new_lines = []
    last_line = [tokens[0]]
    for token in tokens[1:]:
        new_length = sum([len(i) + 1 for i in last_line]) + len(token) + 1
        if new_length > size_limit:
            new_lines.append(last_line)
            last_line = [token]
        else:
            last_line.append(token)
    new_lines.append(last_line)
    return '\n'.join([' '.join(i) for i in new_lines])


def plot_conv(one):
    person_a = one[one['person'] == 'person_a']
    person_b = one[one['person'] == 'person_b']

    fig = plt.figure(figsize=(15, 10))
    # polarity plot settings
    ax1 = fig.add_subplot(131)
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_yticks(range(one.shape[0]))
    ax1.set_title('Polarity')
    ax1.plot(person_a['polarity'], person_a.index, label='person_a')
    ax1.plot(person_b['polarity'], person_b.index, label='person_b')
    ax1.legend()
    for i, act, emo in zip(person_a.index, person_a['act'], person_a['emo']):
        ax1.annotate(act, (person_a.loc[i]['polarity'], i + .2))
        ax1.annotate(emo, (person_a.loc[i]['polarity'], i))

    for i, act, emo in zip(person_b.index, person_b['act'], person_b['emo']):
        ax1.annotate(act, (person_b.loc[i]['polarity'], i + .2))
        ax1.annotate(emo, (person_b.loc[i]['polarity'], i))

    # conversation
    ax2 = fig.add_subplot(132)
    ax2.set_yticks(range(one.shape[0]))
    ax2.set_title('Conversation')
    ax2.set_ylim(-.5, one.shape[0] - .75)
    for i, utter in zip(person_a.index, person_a['utter']):
        ax2.annotate(wrap_text(utter, 40), (0.1, i - .1), bbox={'pad': 3}, wrap=True, horizontalalignment='left')
    for i, utter in zip(person_b.index, person_b['utter']):
        ax2.annotate(wrap_text(utter, 40), (0.1, i - .1), wrap=True, horizontalalignment='left')
    #     ax2.text(0, i, utter)

    # subjectivity plot settings
    ax3 = fig.add_subplot(133)
    ax3.set_xlim(-.05, 1.05)
    ax3.set_yticks(range(one.shape[0]))
    ax3.set_title('Subjectivity')

    # subjectivity plot
    ax3.plot(person_a['subjectivity'], person_a.index)
    ax3.plot(person_b['subjectivity'], person_b.index)


def plot_prec_rec(results: dict, title='', save_name=None, figsize=(5, 5)):
    """Plots precision and recall curves for results dictionary
    Parameters
    ----------
    results : dict
        e.g.
        {'y_true': [np.array, ... ],
         'y_proba': [np.array, ... ],
         'models': [sklearn.model, ... ],
         'classes': ['directive', 'commissive' ... ]}
         `^^^ format of output from data.cv_stratified_shuffle()`
    title : str
        The title for the plot
    save_name : str
        If you want to save this figure
    figsize : tuple (n, n)
        Size of figure

    Returns
    -------
    None
    """
    classes = results['classes']
    fig = plt.figure(1, figsize=figsize)
    ax = fig.add_subplot(111)

    # plot each split
    for y_true, y_proba in zip(results['y_true'], results['y_proba']):
        y_true_cat = pd.get_dummies(y_true)
        # reset colors so classes share the same color
        ax.set_prop_cycle(None)
        for i, cls in enumerate(classes):
            precision, recall, thresholds = \
                precision_recall_curve(y_true_cat.values[:, i],
                                       y_proba[:, i])
            ax.plot(recall, precision, label=cls)

    # show legends matching colors to splits
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # plot mean precision and recalls in black
    for i, cls in enumerate(classes):
        y_true = np.concatenate(results['y_true'])
        y_true_cat = pd.get_dummies(y_true)
        y_proba = np.concatenate(results['y_proba'])
        precision, recall, thresholds = \
            precision_recall_curve(y_true_cat.values[:, i],
                                   y_proba[:, i])
        ax.plot(recall, precision, label=cls, color='black')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    if save_name:
        plt.savefig(save_name)
