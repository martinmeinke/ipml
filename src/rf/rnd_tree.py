from scipy.weave import inline
from scipy.weave import converters
import math
import random
import operator
import numpy as np

'''
function for parallel building of trees
args[0] has the attributes for RandomTree constructor
args[1] is the whole train_data
args[2] are the whole train_labels
args[3] is the data_subset for the tree
args[4] are the attributes
args[5] is the oob_subset (out of bag) for the tree

'''

def parallel_build_tree(args):
    tree = RandomTree(args[0])
    tree.build_tree(args[1], args[2], args[3], args[4], args[5])
    return tree

class RandomTree(object):
    root_node = None
    var_importance = None
    used_features = None
    
    def __init__(self, f_parms):
        self.f_parms = f_parms

    '''
    wrapper to build the tree and set the root node
    '''
    def build_tree(self, data, labels, data_subset, attribute_subset, oob=None):
        actual_depth = 0
        self.root_node = self.generate_tree(data, labels, data_subset, oob, attribute_subset, actual_depth)
        if(oob is not None):
            __ , num_features = np.shape(data)
            #set initial variable importance to zero
            var_importance = np.zeros(num_features)
            used = np.ones(num_features)
            #set the variable importance for the used variables in the tree
            self.root_node.calc_variable_importance(var_importance, used)
            self.var_importance = var_importance
            self.used_features = used
        
        
    '''
    generates a rondomized decision tree
    '''
    def generate_tree(self, train_data, train_labels, data_subset, oob, feature_subset, actual_depth):
        
        default_class = self.get_majority_class(data_subset, train_labels)
        actual_depth +=1
        features = list(feature_subset)
        best_attribute=0
        #Base cases:
        #No more train_data or maximal depth reached, take the majority
        if len(data_subset) < 2 or (actual_depth > self.f_parms.MAX_TREE_DEPTH and self.f_parms.MAX_TREE_DEPTH > 0):
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
        l_child = self.generate_tree(train_data, train_labels, yes_set, oob, feature_subset, actual_depth)
        r_child = self.generate_tree(train_data, train_labels, no_set, oob, feature_subset, actual_depth)
        root.set_l_child(l_child)
        root.set_r_child(r_child)
        if(oob is not None and root.decision_node is not None):
            root.set_gini_importance(self.calc_gini_gain(train_data, train_labels, data_subset, best_attribute, best_threshold))
        return root
    
    def decide(self, features):
        return self.root_node.decide(features)
    
    def calc_thresholds(self, train_data, data_subset, attribute, num_steps):
        minimum, maximum = self.calc_feature_range(train_data, data_subset, attribute)
        vals = []
        for __ in xrange(num_steps):
            vals.append(random.uniform(minimum,maximum))
        return vals
        #step_size = (maximum - minimum) / float(num_steps)
        #return [minimum+step_size * i for i in xrange(1, num_steps)]

    def calc_feature_range(self, train_data, data_subset, attribute):
        max_val = train_data[data_subset[0]][attribute]
        min_val = train_data[data_subset[0]][attribute]
        for item in data_subset:
            if train_data[item][attribute] > max_val:
                max_val = train_data[item][attribute]
            if train_data[item][attribute] < min_val:
                min_val = train_data[item][attribute]
        return (min_val, max_val)
    
    def calc_gini_gain(self, train_data, train_labels, data_subset, attribute, threshold):
        intial_gini = self.calc_gini_index(data_subset, train_labels)
        size = float(len(data_subset))
        yes_set, no_set = self.split_data_by_attribute(train_data, data_subset, attribute, threshold)
        yes_gini = self.calc_gini_index(yes_set, train_labels)
        no_gini = self.calc_gini_index(no_set, train_labels)
        
        return (intial_gini - ((len(yes_set) / size) * yes_gini) - ((len(no_set) / size) * no_gini))
                
    def calc_gini_index(self, data_subset, train_labels):
        frequencies = {}
        gini = 1
    
        for item in data_subset:
            item_class = train_labels[item][0]
            try:
                frequencies[item_class] += 1
            except:
                frequencies[item_class] = 1
                
        size = float(len(data_subset))
        
        for freq in frequencies.values():
            p = freq/size
            gini -= p * p
        return gini
    
    def calc_inf_gain(self, train_data, train_labels, data_subset, attribute, threshold):
        size = float(len(data_subset))
        yes_set, no_set = self.split_data_by_attribute(train_data, data_subset, attribute, threshold)
        return self.calc_entropy(data_subset, train_labels) \
               - ((len(yes_set) / size) * self.calc_entropy(yes_set, train_labels)) \
               - ((len(no_set) / size) * self.calc_entropy(no_set, train_labels))
                    
    def split_data_by_attribute_(self, train_data, data_subset, attribute, threshold):
        yes_set, no_set = [], []
        for item in data_subset:
            if train_data[item][attribute] > threshold:
                yes_set.append(item)
            else:
                no_set.append(item)
        return [yes_set, no_set]
    
    def split_data_by_attribute(self, train_data, data_subset, attribute, threshold):
        split = """
        int i, item;
        float val;
        py::list yes, no;
        py::tuple result(2);
        for(i=0;i<length;i++){
             item = int(data_subset[i]);
             val =  float(train_data[item][attribute]);
             if(val > threshold)
             {
                 yes.append(item);
             }
             else{
                 no.append(item);
             }
        }
        result[0] = yes;
        result[1] = no;
        return_val = result;
        """
        length=len(data_subset)
        return inline(split,['length','train_data','data_subset', 'attribute', 'threshold'],type_converters = converters.blitz, extra_compile_args=['-w'],verbose=1)
    
    
    def calc_entropy_(self, data_subset, train_labels):
        if(len(data_subset) == 0):
            return 0.0
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
            ent -= p * np.log(p)
            
        return ent
    
    def calc_entropy(self, data_subset, train_labels):
        entropy = """
        int i, item_class, item;
        float p1, p2, ent;
        float freq[3] = {0.,0.,0.}; 
        
        for(i=0;i<length;i++){
             item = int(data_subset[i]);
             item_class =  int(train_labels[item][0]) +1;
             freq[item_class] = freq[item_class] +1;
        }
        //p1 = freq[1]/length; //if test
        p1 = freq[0]/length; //if real
        p2 = freq[2]/length;
        if(p1 == 1 || p2 == 1)
        {
            return_val = 0.0;
        }
        else
        {
            return_val = 0.0 - p1*log(p1) - p2*log(p2);
        }
        ent = 0.0 - p1*log(p1) - p2*log(p2);
        """
        length=len(data_subset)
        return inline(entropy,['length','data_subset','train_labels'],type_converters = converters.blitz, extra_compile_args=['-w'])
    
    
    
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
        self.gini = None
        
    def set_gini_importance(self, gini_importance):
        self.gini = gini_importance
        
    def calc_variable_importance(self, features, used):
        if(self.decision_node):
            return 
        #only increase used features if already used
        if(features[self.feature_index] > 0):
            used[self.feature_index] = used[self.feature_index] +1
        features[self.feature_index] = features[self.feature_index] + self.gini
        self.right_child.calc_variable_importance(features, used)
        self.left_child.calc_variable_importance(features, used)

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

        