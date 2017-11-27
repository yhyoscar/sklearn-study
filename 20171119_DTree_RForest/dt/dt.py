from random import seed
from random import randrange

def gini_index(groups, classes):
    # count all samples at a splitting
    n = sum([len(group) for group in groups])
    # get weighted gini index for each group
    gini = 0
    for g in groups:
        size = len(g)
        # avoid divided by 0
        if size == 0:
            continue
        score = 0
        # score the group based on the score for each class
        for c in classes:
            p = [row[-1] for row in g].count(c) / float(size)
            score += p*p
        # weighted group score
        gini += (1 - score) * size / n
    return gini

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))  # 0, 1 in this example
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):  # not including the last ele which is class label
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# dataset = [[2.771244718,1.784783929,0],
# 	[1.728571309,1.169761413,0],
# 	[3.678319846,2.81281357,0],
# 	[3.961043357,2.61995032,0],
# 	[2.999208922,2.209014212,0],
# 	[7.497545867,3.162953546,1],
# 	[9.00220326,3.339047188,1],
# 	[7.444542326,0.476683375,1],
# 	[10.12493903,3.234550982,1],
# 	[6.642287351,3.319983761,1]]
# split = get_split(dataset)
# print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    import pdb
    pdb.set_trace()
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

# tree = build_tree(dataset, 1, 1)
# print(tree)

# make a prediction with DT
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# #  predict with a stump
# stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
# for row in dataset:
#     prediction = predict(stump, row)
#     print('Expected=%d, Got=%d' % (row[-1], prediction))

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)

# Load a txt file
def load_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
    dataset = []
    for l in lines:
        l = l.strip().split(',')
        ele = []
        for e in l:
            ele.append(float(e))
        dataset.append(ele)
    return dataset

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def main():
    # Test CART on Bank Note dataset
    seed(1)
    # load and prepare dataset
    filename = '../data/data_banknote_authentication.txt'
    dataset = load_txt(filename)
    # evaluate algorithm
    n_folds = 5
    max_depth = 5
    min_size = 10
    scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

if __name__ == '__main__':
    main()