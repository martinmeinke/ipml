from numpy import *
import numpy as np
import theano
import theano.tensor as T
from theano.printing import Print as tPrint
from theano.printing import pydotprint as tPngGraph
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse
import logging

DEBUG = False

def tDebugPrint(msg, val):
    return tPrint(msg)(val) if DEBUG else val

def toTheanoBool(x):
    return ifelse(T.ge(x.sum(), 1), 1, 0) 

def selectRandomJ_(i, m):
    rstr = RandomStreams()
    # TODO: make somehow sure that the random integer is not i
    randint = rstr.random_integers(None, 0, m -1, ndim=0)
#    randint = tPrint("Taking random j: ")(randint)
    return randint
    # return T.cast(T.as_tensor_variable(456), 'int64')

def clipAlpha_(aj, H, L):
    aj = ifelse(toTheanoBool(T.gt(aj, H)), H, aj)
    aj = ifelse(toTheanoBool(T.gt(L, aj)), L, aj)
    return aj

"""
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj
"""

def fkKernelTrans(X_, kernel):
    logging.info("Applying full theano kernel transformation to data")

    if kernel[0]!='rbf':
        raise Exception("Kernel type not supported")

    X = T.matrix("X")
    gamma = T.dscalar("gamma")
    
    # we need to transpose the result because the results of the per-row actions are usually columns
    transKernelized = theano.scan(lambda row : kernelTrans_(X, row, gamma), sequences=X)[0].T
    compKernel = theano.function(inputs=[X, gamma], outputs=transKernelized, on_unused_input='ignore')
    
    gamma_ = 1/(-1*kernel[1]**2)
    return compKernel(X_, gamma_)

def kernelTrans(X_, A_, kernel):
    """
    Apply the kernel transformation in dataset X for the features A.
    'kernel' is a tuple with either 'lin' or 'rbf' as the first value and a possible sigma for 'rbf'
    as the second value
    """
    if kernel[0]!='rbf':
        raise Exception("Kernel type not supported")
    
    X = T.matrix("X")
    A = T.matrix("A")
    gamma = T.dscalar("gamma")
    kernelized = kernelTrans_(X, A, gamma)
    
    gamma_ = 1/(-1*kernel[1]**2)
    compKernelized = theano.function(inputs=[X, A, gamma], outputs=kernelized, on_unused_input='ignore')
    return compKernelized(X_, A_, gamma_)

def kernelTrans_(X, A, gamma):
    calcCol = lambda X, A : theano.scan(fn=lambda row, A : ((row - A) ** 2).sum(), sequences=X, non_sequences=A)[0]
    return T.exp(calcCol(X, A) * gamma)

class operatingSymbolStruct:
    def __init__(self, sS):
        self.labels = sS.labels
        self.C = sS.C
        self.tol = sS.tol
        self.K = sS.K
        self.m = sS.m
        self.alphas = sS.alphas
        self.b = sS.b
        self.eCache = sS.eCache
        
    def retlist(self, *args):
        return [self.labels, self.C, self.tol, self.K, self.m, self.alphas, self.b, self.eCache] + list(args)

class symbolStruct:
    def __init__(self, oS):
        self.labels = theano.shared(oS.labelMat, "labels", borrow=True)
        self.C = theano.shared(oS.C, "C")
        self.tol = theano.shared(oS.tol, "tol")
        self.K = theano.shared(oS.K, "K", borrow=True)
        self.m = theano.shared(oS.m, "m")
        self.alphas = theano.shared(oS.alphas, "alphas", borrow=True)
        self.b = theano.shared(oS.b, "b")
        self.eCache = theano.shared(oS.eCache, "eCache", borrow=True)
    
    def _symlist(self):
        return (self.labels, self.C, self.tol, self.K, self.m, self.alphas, self.b, self.eCache)
    
    def updatesOf(self, expression):
        syms = self._symlist()
        numsyms = len(syms)
        return [(syms[i], expression[i]) for i in range(0, numsyms)]
        
    def returnValsOf(self, expression):
        numsyms = len(self._symlist())
        return expression[numsyms:]

    def save(self, oS):
        oS.labelMat = self.labels.get_value()
        oS.C = self.C.get_value()
        oS.tol = self.tol.get_value()
        oS.K = self.K.get_value()
        oS.m = self.m.get_value()
        oS.alphas = self.alphas.get_value()
        oS.b = self.b.get_value()
        oS.eCache = self.eCache.get_value()

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = np.asarray(dataMatIn, dtype=theano.config.floatX)
        self.labelMat = np.asarray(classLabels, dtype=theano.config.floatX)
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = np.mat(zeros((self.m,1), dtype=theano.config.floatX))
        self.b = 0.0
        self.eCache = np.mat(zeros((self.m,2), dtype=theano.config.floatX)) #first column is valid flag

        self.K = mat(fkKernelTrans(self.X, kTup))
        logging.info("K shape: %s", str(self.K.shape))
        
        # Test: Check against old implementation
        # logging.info("Applying old kernel transformation to data")
        # oldK = mat(zeros((self.m,self.m)))
        # for i in range(self.m):
        #     oldK[:,i] = oldSMO.kernelTrans(self.X, self.X[i,:], kTup)
        # logging.info("K shape: %s", str(self.K.shape))
        # logging.info("Old an new K are equal: %s", str(np.allclose(oldK, self.K, atol=10**-5)))

def calcEk_(sS, k):
    # take [0] index as the result is a vector with 0 element
    return (T.dot((sS.alphas * sS.labels).T, sS.K[:,k]) + sS.b - sS.labels[k])[0]

"""
def calcEk(oS, k_):
    sS = symbolStruct()
    k = T.iscalar("k")
    
    Ek = calcEk_(sS, k)
    compEk = theano.function(inputs=[sS.labels, sS.alphas, sS.K, sS.b, k], outputs=Ek)
    res = compEk(oS.labelMat, oS.alphas, oS.K, oS.b, k_)
    logging.debug("E[%s] is %s", str(k_), str(res))
    return res
"""

def selectJ_(sS, i, Ei):
    # make sure error of i is cached
    sS.eCache = T.set_subtensor(sS.eCache[i,:], [1, Ei])
    
    # code to check error cache list for error with biggest delta to Ei
    validEcacheList = sS.eCache[:,0].nonzero()[0]
    
    aDeltaError = lambda k : abs(Ei - calcEk_(sS, tDebugPrint("k is ", k)))
    bDeltaError = lambda k : tDebugPrint("delta is ", aDeltaError(k))
    deltaError = lambda k : ifelse(toTheanoBool(T.eq(i, k)), T.cast(0.0, 'float64'), bDeltaError(k))
    deltaErrors = theano.scan(deltaError, sequences=[validEcacheList])[0]
    maxDeltaErrorJ = validEcacheList[T.argmax(deltaErrors)]
    maxDeltaErrorJAndEj = [calcEk_(sS, maxDeltaErrorJ), maxDeltaErrorJ]
    
    # if we don't have cached errors, yet, we need code to select a random j and Ej
    randomJ = selectRandomJ_(i, sS.m)
    randomJAndError = [calcEk_(sS, randomJ), randomJ]

    # either return a j selected by cached errors or random
    validECacheListLen = validEcacheList.shape[0]
    # validECacheListLen = tPrint("The length of valid cahced errors is: ")(validECacheListLen)
    return ifelse(T.gt(validECacheListLen, 1), maxDeltaErrorJAndEj, randomJAndError)

"""
def selectJ(i_, oS, Ei_):
    sS = symbolStruct()
    Ei = T.scalar()
    i = T.iscalar("i")
    
    selectedJ = selectJ(sS, i, Ei)
    compSelectedJ = theano.function(inputs=[sS.eCache, sS.labels, sS.alphas, sS.K, sS.b, sS.m, i, Ei], outputs=selectedJ, on_unused_input='ignore')
    
    # debugFun = theano.function(inputs=[eCache, Ei, labels, alphas, K, b, i, m], outputs=calcEk_(alphas, labels, K, b, randomJ), on_unused_input='ignore')
    # logging.debug("Value of calcEk_: %s", str(debugFun(oS.eCache, Ei_, oS.labelMat, oS.alphas, oS.K, oS.b, i_, oS.m)))

    oS.eCache[i_,:] = [1, Ei_]    
    return compSelectedJ(oS.eCache, oS.labelMat, oS.alphas, oS.K, oS.b, oS.m, i_, Ei_)

    
def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    logging.debug("Updating eCache[%s,:] to %s", str(k), str(Ek))
    oS.eCache[k,:] = [1,Ek]
"""

def updateEk_(sS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk_(sS, k)
    # logging.debug("Updating eCache[%s,:] to %s", str(k), str(Ek))
    k = tDebugPrint("Update error at index: ", k)
    Ek = tDebugPrint("Updated error to value: ", Ek)
    sS.eCache = T.set_subtensor(sS.eCache[k,:], [1,Ek])


# theano implementation of innerL is split in several functions to support early exit and keep order
def innerL_(sS, i):
    Ei = calcEk_(sS, i)
    
    # use "+" instead of "or" and "*" instead of "and"
    checkUselessAlpha1 = T.ge(sS.labels[i] * Ei, -sS.tol) + T.ge(sS.alphas[i], sS.C)
    checkUselessAlpha2 = T.le(sS.labels[i]*Ei, sS.tol) + T.lt(sS.alphas[i], 0)
    isUselessAlpha = toTheanoBool(checkUselessAlpha1 * checkUselessAlpha2)
    
    updateL = innerL_alphaInRange_(sS, i, Ei)
    earlyret = sS.retlist(0)
    return ifelse(isUselessAlpha, earlyret, updateL)

def innerL_alphaInRange_(sS, i, Ei):
    
    Ej, j = selectJ_(sS, i, Ei)

    ijAreEqualClass = toTheanoBool(T.eq(sS.labels[i], sS.labels[j]))
    L = T.maximum(0,    ifelse(ijAreEqualClass,   sS.alphas[j] + sS.alphas[i] - sS.C,    sS.alphas[j] - sS.alphas[i]))
    H = T.minimum(sS.C, ifelse(ijAreEqualClass,   sS.alphas[j] + sS.alphas[i],           sS.C + sS.alphas[j] - sS.alphas[i]))

    eta = 2.0 * sS.K[i,j] - sS.K[i,i] - sS.K[j,j]
    hasBadEtaAndRanges = toTheanoBool(T.eq(L, H) + T.ge(eta, 0))
    
    updateL = innerL_updateAlphaWithEta_(sS, i, Ei, j, Ej, H, L, eta)
    earlyret = sS.retlist(0)  
    #logging.debug("earlyr: type %s, len %s, vals %s", str(type(updateL)), str(len(updateL)), str(updateL))
    #logging.debug("update: type %s, len %s, vals %s", str(type(earlyret)), str(len(earlyret)), str(earlyret))
    return ifelse(hasBadEtaAndRanges, earlyret, updateL)

def innerL_updateAlphaWithEta_(sS, i, Ei, j, Ej, H, L, eta):
    alphaIold = sS.alphas[i].copy()
    alphaJold = sS.alphas[j].copy()

    # logging.debug("j is %s, Ej is %s, alphas[j] shape: %s, update shape: %s", str(j), str(Ej), str(np.shape(oS.alphas[j])), str(np.shape(update)))
    # update alpha
    H = tDebugPrint("H= ", H)
    L = tDebugPrint("L= ", L)
    Ei = tDebugPrint("Ei= ", Ei)
    Ej = tDebugPrint("Ej= ", Ej)
    eta = tDebugPrint("eta= ", eta)
    
    updatedAlpha = sS.alphas[j] - (sS.labels[j] * (Ei - Ej) / eta)
    updatedAlpha = clipAlpha_(updatedAlpha, H, L)
    j = tDebugPrint("Updating alpha ", j)
    updatedAlpha = tDebugPrint("Setting alpha to", updatedAlpha)
    sS.alphas = T.set_subtensor(sS.alphas[j], updatedAlpha)
    updateEk_(sS, j) # add error for alpha[j]
   
    alphaJHasntChanged = toTheanoBool(T.lt(T.abs_(sS.alphas[j] - alphaJold), 0.00001))
    # alphaJHasntChanged = tPrint("j not moving enough:")(alphaJHasntChanged)
    
    updateL = innerL_updateAlphaAndB_(sS, i, Ei, j, Ej, alphaIold, alphaJold)
    earlyret = sS.retlist(0)
    #logging.debug("earlyr: type %s, len %s, vals %s", str(type(updateL)), str(len(updateL)), str(updateL))
    #logging.debug("update: type %s, len %s, vals %s", str(type(earlyret)), str(len(earlyret)), str(earlyret))
    return ifelse(alphaJHasntChanged, earlyret, updateL)

def innerL_updateAlphaAndB_(sS, i, Ei, j, Ej, alphaIold, alphaJold):
    # update alphas[i] by the same amount as alphas[j]
    sS.alphas = T.set_subtensor(sS.alphas[i], sS.alphas[i] + (sS.labels[j]*sS.labels[i]*(alphaJold - sS.alphas[j])))
    # update error for new alpha[i]
    updateEk_(sS, i)
    
    # update b
    b1 = sS.b - Ei- sS.labels[i] * (sS.alphas[i] - alphaIold) * sS.K[i,i] - sS.labels[j] * (sS.alphas[j] - alphaJold) * sS.K[i,j]
    b2 = sS.b - Ej- sS.labels[i] * (sS.alphas[i] - alphaIold) * sS.K[i,j] - sS.labels[j] * (sS.alphas[j] - alphaJold) * sS.K[j,j]
    
    # confitions: use * instead of "and" and + instead of "or"
    alphaInRange = lambda x : toTheanoBool(T.gt(sS.alphas[x], 0) * T.lt(sS.alphas[x], sS.C))
    checkForB2 = ifelse(alphaInRange(j), b2, (b1 + b2)/2.0 )
    sS.b = ifelse(alphaInRange(j), b1, checkForB2)[0] # 0 index because the values are in form of a one element vector
    sS.b = tDebugPrint("b= ", sS.b)
    
    return sS.retlist(1)

# end of theano innerL implementation



def innerL(i_, oS):
    sS = symbolStruct(oS)
    oSS = operatingSymbolStruct(sS)
    i = T.iscalar("i")
    
    innerLUpdate = innerL_(oSS, i)
    # theano.pp(innerLUpdate[0])
    logging.info("Compiling innerL theano function")
    updates = sS.updatesOf(innerLUpdate)
    outputs = sS.returnValsOf(innerLUpdate)
    logging.debug("updates of innerL: %s", str(updates))
    logging.debug("outputs of innerL: %s", str(outputs))
    compInnerL = theano.function([i], outputs, updates=updates, on_unused_input='ignore')
    logging.info("Compilation finished")
    
    logging.debug("Args to innerL: %s", str([i_]))
    res = compInnerL(i_)
    # save result from shared to oS
    sS.save(oS)
       
    return res[0]
    
    

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO

    theano.config.mode = "FAST_RUN"    
    theano.config.linker = "cvm"

    
    oS = optStruct(dataMatIn,classLabels,C,toler, kTup)
    sS = symbolStruct(oS)
    oSS = operatingSymbolStruct(sS)
    i_ = T.iscalar("i")
    
    
    innerLUpdate = innerL_(oSS, i_)
    logging.info("Compiling innerL theano function")
    updates = sS.updatesOf(innerLUpdate)
    outputs = sS.returnValsOf(innerLUpdate)
    logging.debug("updates of innerL: %s", str(updates))
    logging.debug("outputs of innerL: %s", str(outputs))
    compInnerL = theano.function([i_], outputs, updates=updates, on_unused_input='ignore')
    logging.info("Compilation finished")
    
    iteration = 0
    entireSet = True; alphaPairsChanged = 0
    logging.info("Starting main loop")
    entireSetRange = range(oS.m)
    while iteration < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        bounds = entireSetRange if entireSet else nonzero((sS.alphas.get_value() > 0) * (sS.alphas.get_value() < C))[0]
        setName = "fullSet" if entireSet else "non-bound"
        runI = 0
        logging.info("This iteration will consist of %d steps", len(bounds))
        for i in bounds:
            alphaPairsChanged += compInnerL(i)[0]
            if runI > 0 and runI % 100 == 0:
                logging.info("%s, iteration: %d i:%d, pairs changed %d" % (setName, iteration, runI, alphaPairsChanged))
            runI += 1
        
        logging.info("Finished %s, iteration number: %d, pairs changed %d" % (setName, iteration, alphaPairsChanged))
        iteration += 1
        if entireSet:
            entireSet = False #toggle entire set loop
        elif alphaPairsChanged == 0:
            entireSet = True
    return sS.b.get_value(), sS.alphas.get_value()
