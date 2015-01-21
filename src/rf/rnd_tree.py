import math
import random
import operator


def parallel_build_tree(args):
    tree = RandomTree(args[0][0],args[0][1],args[0][2],args[0][3])
    tree.build_tree(args[1], args[2])
    return tree


class RandomTree(object):
    root_node = None
    
    def __init__(self, min_gain, num_attr, num_thres, max_tries):
        self.MIN_GAIN = min_gain
        self.NUM_ATTRIBUTES = num_attr
        self.NUM_THRES_STEPS = num_thres
        self.MAX_TRIES = max_tries

    '''
    wrapper to build the tree and set the root node
    '''
    def build_tree(self, data, attributes):
        self.root_node = self.generate_tree(data, attributes)
        
    '''
    generates a rondomized decision tree
    '''
    def generate_tree(self, data, attributes):
        default_val = self.get_majority_class(data)
    
        #Base cases:
        #No more data, take the majority
        if len(data) < 2:
            root = DecisionTreeNode()
            root.set_label(default_val)
            return root
    
        #All data points have same label
        firstValue = data.values()[0]
        sameValues = True
        for value in data.values():
            if firstValue != value:
                sameValues = False
        if sameValues:
            root = DecisionTreeNode()
            root.set_label(firstValue)
            return root
    
        #decide best feature to split on
        
        attributes = self.list_index_sub_sample(attributes, self.NUM_ATTRIBUTES)
        best_gain = -1
        best_threshold = -1
        
        for i in range(self.MAX_TRIES):
            for attribute in attributes:
                for threshold in self.calc_thresholds(self.find_max(data,attribute), self.NUM_THRES_STEPS):
                    gain = self.calc_inf_gain(data, attribute, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_attribute = attribute
                        best_threshold = threshold
            
            #if gain sufficient break
            if(best_gain > self.MIN_GAIN):
                break
            #if maximum iteration not reached try new set of attributes
            elif best_gain < self.MIN_GAIN and i < (self.MAX_TRIES -1):
                attributes = self.list_index_sub_sample(attributes, self.NUM_ATTRIBUTES)
            else:
                root = DecisionTreeNode()
                root.set_label(default_val)
                return root
    
        #Recursive build tree
        root = DecisionTreeNode(best_attribute, best_threshold)
        yes_set, no_set = self.split_data_by_attribute(data, best_attribute, best_threshold)
        l_child = self.generate_tree(yes_set, attributes)
        r_child = self.generate_tree(no_set, attributes)
        root.set_l_child(l_child)
        root.set_r_child(r_child)
    
        return root
    
    def decide(self, features):
        return self.root_node.decide(features)
    
    def calc_thresholds(self, maximum, num_steps):
        step_size = maximum / float(num_steps)
        return [step_size * i for i in xrange(1, num_steps)]
    
    def find_max(self, data, attribute):
        max_value = -1
        for elem in data.items():
            if elem[0][attribute] > max_value:
                max_value = elem[0][attribute]
        return max_value
    
    def calc_inf_gain(self, data, attribute, threshold):
        size = float(len(data))
        yes_set, no_set = self.split_data_by_attribute(data, attribute, threshold)
        return self.calc_entropy(data) \
               - ((len(yes_set) / size) * self.calc_entropy(yes_set)) \
               - ((len(no_set) / size) * self.calc_entropy(no_set))
    
    def calc_entropy(self, data):
        frequencies = {}
        ent = 0.0
        values = data.values()
    
        for i in xrange(len(values)):
            if frequencies.has_key(values[i]):
                frequencies[values[i]] += 1
            else:
                frequencies[values[i]] = 1
    
        for freq in frequencies.values():
            p = freq/float(len(values))
            ent -= p * math.log(p, 2)
            
        return ent
    
    def get_majority_class(self, data):
        counts = {}
        for value in data.values():
            if counts.has_key(value):
                counts[value] += 1
            else:
                counts[value] = 1
        sortedCounts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        return sortedCounts[0][0]    
    
    def split_data_by_attribute(self, data, attribute, threshold):
        yes_set, no_set = {}, {}
        for datum in data.items():
            if datum[0][attribute] > threshold:
                yes_set[datum[0]] = datum[1]
            else:
                no_set[datum[0]] = datum[1]
        return [yes_set, no_set]
    
    def list_index_sub_sample(self, orig_list, sub_size):
        subset = []
        for _ in xrange(sub_size):
            subset.append(random.randrange(0, len(orig_list)))
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

        