import math
import random
import operator

class RandomTree(object):
    MIN_GAIN = 10e-5
    
    root_node = None
    
    def __init__(self, data, attributes):
        self.root_node = self.generateTree(data, attributes)


    def generateTree(self, data, attributes):
        """
        Returns a randomized decision tree
        """
        defaultValue = self.getMajority(data)
    
        #Base cases:
        #No more data, take the majority
        if len(data) < 2:
            root = DecisionTreeNode()
            root.setLabel(defaultValue)
            return root
    
        #All data points have same label
        firstValue = data.values()[0]
        sameValues = True
        for value in data.values():
            if firstValue != value:
                sameValues = False
        if sameValues:
            root = DecisionTreeNode()
            root.setLabel(firstValue)
            return root
    
        #Greedily decide best feature to split on
        attributes = self.listIndexSubSample(attributes, 8)
    
        best_gain = -1
        best_threshold = -1
        for attribute in attributes:
            for threshold in self.getThresholds(self.findMax(data,attribute), 10):
                gain = self.informationGain(data, attribute, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = attribute
                    best_threshold = threshold
    
        if best_gain < self.MIN_GAIN:
            root = DecisionTreeNode()
            root.setLabel(defaultValue)
            return root
    
        #Recursive build tree
        root = DecisionTreeNode(best_attribute, best_threshold)
        yes_set, no_set = self.splitDataByAttribute(data, best_attribute, best_threshold)
        yes = self.generateTree(yes_set, attributes)
        no = self.generateTree(no_set, attributes)
        root.setYes(yes)
        root.setNo(no)
    
        return root
    
    def decide(self, features):
        return self.root_node.decide(features)
    
    
    def getThresholds(self, maximum, num_steps):
        step_size = maximum / float(num_steps)
        return [step_size * i for i in xrange(1, num_steps)]
    
    def findMax(self, data, attribute):
        max_value = -1
        for elem in data.items():
            if elem[0][attribute] > max_value:
                max_value = elem[0][attribute]
        return max_value
    
    def informationGain(self, data, attribute, threshold):
        size = float(len(data))
        yes_set, no_set = self.splitDataByAttribute(data, attribute, threshold)
        return self.entropy(data) \
               - ((len(yes_set) / size) * self.entropy(yes_set)) \
               - ((len(no_set) / size) * self.entropy(no_set))
    
    def entropy(self, data):
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
    
    def getMajority(self, data):
        counts = {}
        for value in data.values():
            if counts.has_key(value):
                counts[value] += 1
            else:
                counts[value] = 1
        sortedCounts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        return sortedCounts[0][0]
    
    
    
    def splitDataByAttribute(self, data, attribute, threshold):
        yes_set, no_set = {}, {}
        for datum in data.items():
            if datum[0][attribute] > threshold:
                yes_set[datum[0]] = datum[1]
            else:
                no_set[datum[0]] = datum[1]
        return [yes_set, no_set]
    
    
    def listIndexSubSample(self, orig_list, sub_size):
        subset = []
        for _ in xrange(sub_size):
            subset.append(random.randrange(0, len(orig_list)))
        return subset

class DecisionTreeNode:
    """
    A binary decision tree
    """

    def __init__(self, feature_index=None, feature_threshold=None):
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.yes = None
        self.no = None
        self.label = None
        self.decisionNode = False

    def setYes(self, yes):
        self.yes = yes

    def setNo(self, no):
        self.no = no

    def setLabel(self, label):
        self.label = label
        self.decisionNode = True

    def decide(self, features):
        if self.decisionNode:
            return self.label
        if features[self.feature_index] > self.feature_threshold:
            return self.yes.decide(features)
        return self.no.decide(features)

        