import numpy as np


def all_words(dim, order):
    """
    Returns a list of tuples containing all words up to :param order: that can be made with :param dim: letters

    :param dim:     Dimension of path
    :param order:   Order to take up to
    :return:        All words
    """

    if dim <= 0 or order <= 0:
        return []

    def backtrack(word, letters):
        if len(word) <= order:
            words.append(word)
        else:
            return

        for letter in letters:
            if letter <= dim:
                backtrack(word + str(letter), letters)

    words = []
    letters = list(range(1, dim + 1))
    backtrack("", letters)
    return sorted(words, key=lambda x: (len(x), x))


def tuples_to_strings(lst):
    return [''.join(map(str, t)) for t in lst]


def shuffle_strings(a, b):
    if a == "": return [b]
    if b == "": return [a]
    return [a[0] + s for s in shuffle_strings(a[1:], b)] + [b[0] + s for s in shuffle_strings(a, b[1:])]


def find_indices(a, index_map):
    return [index_map[string] for string in a]


def shuffle_product(i: int, j: int, words: list) -> np.ndarray:
    """
    Computes the shuffle product between two basis vectors with dimension given by the dimension and order of the
    corresponding signature.

    :param i:               Coordinate of 1 in the first basis vector
    :param j:               Coordinate of 1 in the second basis vector
    :param words:           Dictionary of words

    :return:                Shuffle product between e_i and e_j
    """

    # Vectors to words
    word_map = {string: index for index, string in enumerate(words)}

    # Define output vector
    res = np.zeros(len(words))

    # Basis vectors to words
    a_letters, b_letters = words[i], words[j]

    # Get shuffles
    shuffles = shuffle_strings(a_letters, b_letters)

    # Represent shuffles as array
    locs = find_indices(shuffles, word_map)

    for ii in locs:
        res[ii] += 1

    return res
