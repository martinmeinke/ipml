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
    def generate_tree(self, train_data, train_labels, data_subset, attribute_subset, actual_depth):
        default_class = self.get_majority_class(data_subset, train_labels)
        actual_depth +=1
    
        #Base cases:
        #No more train_data or maximal depth reached, take the majority
        if len(data_subset) < 2 or actual_depth > self.f_parms.MAX_TREE_DEPTH:
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
    
        #decide best feature to split on
        
        attributes = self.list_index_sub_sample(attribute_subset, self.f_parms.NUM_ATTRIBUTES)
        best_gain = -1
        best_threshold = -1
        
        for i in range(self.f_parms.MAX_TRIES):
            for attribute in attributes:
                for threshold in self.calc_thresholds(self.find_max(train_data, data_subset, attribute), self.f_parms.NUM_THRES_STEPS):
                    gain = self.calc_inf_gain(train_data, train_labels, data_subset, attribute, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_attribute = attribute
                        best_threshold = threshold
                
            #if no minimum gain set or gain sufficient break
            if(self.f_parms.MIN_GAIN == None or best_gain > self.f_parms.MIN_GAIN):
                break
            #if maximum iteration not reached try new set of attributes
            elif best_gain < self.f_parms.MIN_GAIN and i < (self.f_parms.MAX_TRIES -1):
                attributes = self.list_index_sub_sample(attribute_subset, self.f_parms.NUM_ATTRIBUTES)
            else:
                root = DecisionTreeNode()
                root.set_label(default_class)
                return root
    
        #Recursive build tree
        root = DecisionTreeNode(best_attribute, best_threshold)
        yes_set, no_set = self.split_data_by_attribute(train_data, data_subset, best_attribute, best_threshold)
        l_child = self.generate_tree(train_data, train_labels, yes_set, attribute_subset, actual_depth)
        r_child = self.generate_tree(train_data, train_labels, no_set, attribute_subset, actual_depth)
        root.set_l_child(l_child)
        root.set_r_child(r_child)
        return root
    
    def decide(self, features):
        return self.root_node.decide(features)
    
    def calc_thresholds(self, maximum, num_steps):
        step_size = maximum / float(num_steps)
        return [step_size * i for i in xrange(1, num_steps)]
    
    def find_max(self, train_data, data_subset, attribute):
        max_value = -1
        for item in data_subset:
            if train_data[item][attribute] > max_value:
                max_value = train_data[item][attribute]
        return max_value
    
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
    
        for freq in frequencies.values():
            p = freq/float(len(data_subset))
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
    
    def list_index_sub_sample(self, orig_list, sub_size):
        subset = []
        for _ in xrange(sub_size):
            subset.append(random.randrange(0, len(orig_list)-1))
        return subset

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

        