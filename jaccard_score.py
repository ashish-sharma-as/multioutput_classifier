from sklearn.metrics import jaccard_score
import numpy as np

a1 = np.array([[1, 0],
               [1, 1]])

a2 = np.array([[0, 1],
               [1, 0]])

js_samples = jaccard_score(a1, a2, average='samples')

js_weighted = jaccard_score(a1, a2, average='weighted')


def jaccard_score_manual(y_true, y_pred):
    # Compute jaccard score
    # np.min element wise will pick up 0 if 0 and 1 are the 2 elements
    # in the 2 arrays being compared, which means no element was intersecting
    # np.min will pick 1 if 1 and 1 are the 2 elements in the 2 arrays
    # which means both elements were 1 and were intersecting
    # HOWEVER, if both elements are 0 and 0, np.min will pick 0,
    # which is correct, but, if the arrays are made of only 1 element
    # each then the jaccard score is 0. But then again, we use jaccard
    # score with multilabel classification and it is fair to assume that
    # the 2 arrays being compared will not have only 1 element each.

    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true,
                                                                  y_pred).sum(
        axis=1)
    print(jaccard)
    print(np.minimum(y_true, y_pred).sum(axis=1))
    print(np.minimum(y_true, y_pred).sum(axis=0))
    return jaccard.mean()



# Call
js_samples_formula = jaccard_score_manual(a1, a2)

# -------------------------------------------------------
# NOTES:
# -------------------
# Average = "samples"
# -------------------
# js_samples = np.minimum(a1, a2) (element-wise) /
#              np.maximum(a1, a2) (element-wise)
#              and then mean() of the js_samples

# min and max of the 2 arrays
# min part: [[0, 0],         max part: [[1, 1],
#            [1, 0]]                    [1, 1]]

# Then sum() above along each row
# min part: [0 1]            max_part: [2 2]

# Division result (element wise) of the 2 arrays above:
# [0 0.5]

# Then take mean of the above vector = (0 + 0.5)/2 = 0.25

# -------------------
# Average = "weighted"
# -------------------
# Instead of taking the mean, weight the 2 in the final step
# 1 part for 0 and 2 parts for 0.5 (so 3 parts in total)
# (1/3)*0 + (2/3)*0.5 = 0.3333333

# -------------------------------------------------------


# Trying with a larger set

a3 = np.array([[1, 0],
               [1, 1],
               [1, 1]])

a4 = np.array([[0, 1],
               [1, 0],
               [1, 0]])

js_samples34 = jaccard_score(a3, a4, average='samples')

js_weighted34 = jaccard_score(a3, a4, average='weighted')

print('end')

# -------------------------------------------------------
