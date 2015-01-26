import math
import random
import operator

'''
function for parallel building of trees
args[0] has the attributes for RandomTree constructor
args[1] is the whole train_data
args[2] are the whole train_labels
args[3] is the data_subset for the tree
args[4] are the attributes
'''

def parallel_build_tree(args):
    tree = RandomTree(args[0])
    tree.build_tree(args[1], args[2], args[3], args[4])
    return tree


class RandomTree(object):
    root_node = None
    
    def __init__(self, f_parms):
        self.f_parms = f_parms

    '''
    wrapper to build the tree and set the root node
    '''
    def build_tree(self, data, labels, data_subset, attribute_subset):
        actual_depth = 0
        self.root_node = self.generate_tree(data, labels, data_subset, attribute_subset, actual_depth)
        
    '''
    generates a rondomized decision tree
    '''
    def generate_tree(self, train_data, train_labels, data_subset, feature_subset, actual_depth):
        default_class = self.get_majority_class(data_subset, train_labels)
        actual_depth +=1
        features = list(feature_subset)
        
        #Base cases:
        #No more train_data or maximal depth reached, take the majority
        if len(data_subset) < 2 or (actual_depth > self.f_parms.MAX_TREE_DEPTH and self.f_parms.MAX_TREE_DEPTH is not None):
            root = DecisionTreeNode()
            root.set_label(default_class)
            return root
    
        #All train_data points have same label
        first_class = train_labels[data_subset[0]][0]
        sameValues = True
        for item in data_subset:
            if first_class != train_labels[item][0]:
                sameValues = False
                break
        if sameValues:
            root = DecisionTreeNode()
            root.set_label(first_class)
            return root
    
        #shuffle all features for randomly picking one
        random.shuffle(features)
        attribute = features.pop()
        best_gain = -1
        best_threshold = -1
        
        #decide best feature to split on
        for i in range(self.f_parms.MAX_TRIES):
            for threshold in self.calc_thresholds(train_data, data_subset, attribute, self.f_parms.NUM_THRES_STEPS):
                gain = self.calc_inf_gain(train_data, train_labels, data_subset, attribute, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = attribute
                    best_threshold = threshold
                
            #if we tried all features to find the best split we can break
            if len(features) < 1:
                break
            #if maximum iteration not reached or no mingain set, try new set of attributes
            elif (self.f_parms.MIN_GAIN < 0 or best_gain < self.f_parms.MIN_GAIN) and i < (self.f_parms.MAX_TRIES -1):
                attribute = features.pop()
            #if no minimum gain set or gain sufficient break
            elif (best_gain > self.f_parms.MIN_GAIN and self.f_parms.MIN_GAIN > 0) or self.f_parms.MIN_GAIN < 0:
                break
            else:
                root = DecisionTreeNode()
                root.set_label(default_class)
                return root
    
        #Recursive build tree
        yes_set, no_set = self.split_data_by_attribute(train_data, data_subset, best_attribute, best_threshold)
        #print "depth=%d  :  split len_left=%d  len_right=%d" %(actual_depth, len(yes_set), len(no_set))
        
        if len(yes_set) < 1 or len(no_set) < 1:
            root = DecisionTreeNode()
            root.set_label(default_class)
            return root
        root = DecisionTreeNode(best_attribute, best_threshold)
        l_child = self.generate_tree(train_data, train_labels, yes_set, feature_subset, actual_depth)
        r_child = self.generate_tree(train_data, train_labels, no_set, feature_subset, actual_depth)
        root.set_l_child(l_child)
        root.set_r_child(r_child)
        return root
    
    def decide(self, features):
        return self.root_node.decide(features)
    
    def calc_thresholds(self, train_data, data_subset, attribute, num_steps):
        minimum, maximum = self.calc_feature_range(train_data, data_subset, attribute)
        step_size = (maximum - minimum) / float(num_steps)
        return [minimum+step_size * i for i in xrange(1, num_steps)]
    
    def calc_feature_range(self, train_data, data_subset, attribute):
        max_val = train_data[data_subset[0]][attribute]
        min_val = train_data[data_subset[0]][attribute]
        for item in data_subset:
            if train_data[item][attribute] > max_val:
                max_val = train_data[item][attribute]
            if train_data[item][attribute] < min_val:
                min_val = train_data[item][attribute]
        return (min_val, max_val)
    
    def calc_inf_gain(self, train_data, train_labels, data_subset, attribute, threshold):
        size = float(len(data_subset))
        yes_set, no_set = self.split_data_by_attribute(train_data, data_subset, attribute, threshold)
        return self.calc_entropy(data_subset, train_labels) \
               - ((len(yes_set) / size) * self.calc_entropy(yes_set, train_labels)) \
               - ((len(no_set) / size) * self.calc_entropy(no_set, train_labels))
               
    def split_data_by_attribute(self, train_data, data_subset, attribute, threshold):
        yes_set, no_set = [], []
        for item in data_subset:
            if train_data[item][attribute] > threshold:
                yes_set.append(item)
            else:
                no_set.append(item)
        return [yes_set, no_set]
    
    def calc_entropy(self, data_subset, train_labels):
        frequencies = {}
        ent = 0.0
    
        for item in data_subset:
            item_class = train_labels[item][0]
            try:
                frequencies[item_class] += 1
            except:
                frequencies[item_class] = 1
                
        size = float(len(data_subset))
        for freq in frequencies.values():
            p = freq/size
            ent -= p * math.log(p, 2)
            
        return ent
    
    def get_majority_class(self, data_subset, train_labels):
        counts = {}
        for item in data_subset:
            item_class = train_labels[item][0]
            try:
                counts[item_class] += 1
            except:
                counts[item_class] = 1
        sortedCounts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        return sortedCounts[0][0]    
    
class DecisionTreeNode:

    def __init__(self, feature_index=None, feature_threshold=None):
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.left_child = None
        self.right_child = None
        self.label = None
        self.decision_node = False

    def set_l_child(self, child_node):
        self.left_child = child_node

    def set_r_child(self, child_node):
        self.right_child = child_node

    def set_label(self, label):
        self.label = label
        self.decision_node = True

    def decide(self, features):
        if self.decision_node:
            return self.label
        #if true left child otherwise right child
        if features[self.feature_index] > self.feature_threshold:
            return self.left_child.decide(features)
        return self.right_child.decide(features)

        