from numpy import *
import numpy as np
import theano
import theano.tensor as T
from theano.printing import Print as tPrint
from theano.printing import pydotprint as tPngGraph
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse
import logging
import smo as oldSMO

def toTheanoBool(x):
    return ifelse(T.ge(x.sum(), 1), 1, 0) 

def colToRow(x):
    # dimshuffle converts the column vector into a row vector
    return x.dimshuffle('x', 0)

def selectRandomJ_(i, m):
    rstr = RandomStreams()
    # TODO: make somehow sure that the random integer is not i
    return rstr.random_integers(None, 0, m, ndim=0)

def clipAlpha_(aj, H, L):
    aj = ifelse(toTheanoBool(T.gt(aj, H)), H, aj)
    aj = ifelse(toTheanoBool(T.gt(L, aj)), L, aj)
    return aj

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def fkKernelTrans(X_, kernel):
    logging.info("Applying full theano kernel transformation to data")
    gamma_ = 1/(-1*kernel[1]**2)
    if kernel[0]!='rbf':
        raise Exception("Kernel type not supported")

    X = T.matrix("X")
    gamma = T.dscalar("gamma")
    
    calcCol = lambda A: theano.scan(fn=lambda row, A : ((row - A) ** 2).sum(), sequences=X, non_sequences=A)[0]
    colKernel = lambda A : T.exp(calcCol(A)*gamma)
    # we need to transpose the result because the results of the per-row actions are usually columns
    transKernelized = theano.scan(lambda row : colKernel(row), sequences=X)[0].T

    # TEST: make sure the shape is (2,), because we computed one value per row based on the row vector A
    # tA = T.row()
    # testEvalArgs = {tA:np.asarray([1,2,3]).reshape(1,3), X:np.arange(0,6).reshape(2,3), gamma:gamma_}
    # testEvaled = colKernel(tA).eval(testEvalArgs)
    # logging.info("Test evaled type: %s,  shape: %s, value: %s", str(type(testEvaled)), str(testEvaled.shape), str(testEvaled))

    compKernel = theano.function(inputs=[X, gamma], outputs=transKernelized, on_unused_input='ignore')
    return compKernel(X_, gamma_)

def kernelTrans(X_, A_, kernel):
    """
    Apply the kernel transformation in dataset X for the features A.
    'kernel' is a tuple with either 'lin' or 'rbf' as the first value and a possible sigma for 'rbf'
    as the second value
    """
    # m,n = shape(X)
    # K = mat(zeros((m,1)))
    X = T.matrix("X")
    A = T.row("A")
    gamma = T.dscalar("sigma")
    # linear
    if kernel[0]=='lin':
        gamma_ = 0
        kernelExp = theano.dot(X, A.T)
    # radial basis kernel
    elif kernel[0]=='rbf':
        gamma_ = 1/-1*kernel[1]**2
        K, _ = theano.scan(fn=lambda row, A : ((row - A) ** 2).sum(), sequences=X, non_sequences=A)
#        for j in range(m):
#            deltaRow = X[j,:] - A
#            K[j] = deltaRow*deltaRow.T
        kernelExp = T.exp(K*gamma)
        # return exp(K/(-1*kernel[1]**2)) # divide is element-wise        
    # unknown
    else:
        raise ValueError("The kernel '%' is unknown" % kernel[0])
    compKernel = theano.function(inputs=[X, A, gamma], outputs=kernelExp, on_unused_input='ignore')
    return mat(compKernel(X_, A_, gamma_)).T

class symbolStruct:
    def __init__(self):
        self.labels = T.col("labels")
        self.C = T.scalar("C")
        self.tol = T.scalar("tol")
        self.K = T.matrix("K")
        self.m = T.iscalar("m")
        self.alphas = T.col("alphas")
        self.b = T.scalar("b")
        self.eCache = T.matrix("eCache")
    
    def symlist(self):
        return [self.labels, self.C, self.tol, self.K, self.m, self.alphas, self.b, self.eCache]
      
    def arglist(self, oS):
        return [oS.labelMat, oS.C,   oS.tol,   oS.K,   oS.m,   oS.alphas,   oS.b,   oS.eCache]
    
    def retlist(self, *args):
        return self.symlist() + [T.as_tensor_variable(x) for x in args]
    
    def saveResults(self, oS, results):
        numargs = len(self.arglist(oS))
        oS.labelMat, oS.C, oS.tol, oS.K, oS.m, oS.alphas, oS.b, oS.eCache = results[:numargs]
        return results[numargs:]

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = np.mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(zeros((self.m,2))) #first column is valid flag

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
    return (T.dot(colToRow(sS.alphas * sS.labels), sS.K[:,k]) + sS.b - sS.labels[k])[0]

def calcEk(oS, k_):
    sS = symbolStruct()
    k = T.iscalar("k")
    
    Ek = calcEk_(sS, k)
    compEk = theano.function(inputs=[sS.labels, sS.alphas, sS.K, sS.b, k], outputs=Ek)
    res = compEk(oS.labelMat, oS.alphas, oS.K, oS.b, k_)
    logging.debug("E[%s] is %s", str(k_), str(res))
    return res

def selectJ_(sS, i, Ei):
    # make sure error of i is cached
    sS.eCache = T.set_subtensor(sS.eCache[i,:], [1, Ei], inplace=True)

    # code to check error cache list for error with biggest delta to Ei
    validEcacheList = sS.eCache[:,0].nonzero()[0]
    deltaErrors = theano.scan(lambda k: abs(Ei - calcEk_(sS, k)), sequences=[validEcacheList])[0]
    selectMaxError = T.max_and_argmax(deltaErrors)
    
    # if we don't have cached errors, yet, we need code to select a random j and Ej
    randomJ = selectRandomJ_(i, sS.m)
    randomJAndError = [calcEk_(sS, randomJ), randomJ]

    # either return a j selected by cached errors or random
    return ifelse(T.gt(validEcacheList.shape[0], 1), selectMaxError, randomJAndError)

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
    # use T.set_subtensor(oS.eCache[k,:], [1,Ek], inplace=True)
    Ek = calcEk(oS, k)
    logging.debug("Updating eCache[%s,:] to %s", str(k), str(Ek))
    oS.eCache[k,:] = [1,Ek]

def updateEk_(sS, k):#after any alpha has changed update the new value in the cache
    # use T.set_subtensor(oS.eCache[k,:], [1,Ek], inplace=True)
    Ek = calcEk_(sS, k)
    # logging.debug("Updating eCache[%s,:] to %s", str(k), str(Ek))
    T.set_subtensor(sS.eCache[k,:], [1,Ek], inplace=True)


# theano implementation of innerL is split in several functions to support early exit and keep order
def innerL_(sS, i):
    Ei = calcEk_(sS, i)
    
    # use "+" instead of "or" and "*" instead of "and"
    checkUselessAlpha1 = T.ge(sS.labels[i] * Ei, -sS.tol) + T.ge(sS.alphas[i], sS.C)
    checkUselessAlpha2 = T.le(sS.labels[i]*Ei, sS.tol) + T.lt(sS.alphas[i], 0)
    
    updateL = innerL_alphaInRange_(sS, i, Ei)
    earlyret = sS.retlist(0)
    return ifelse(toTheanoBool(checkUselessAlpha1 * checkUselessAlpha2), earlyret, updateL)

def innerL_alphaInRange_(sS, i, Ei):
    Ej, j = selectJ_(sS, i, Ei) # both return values are ndarrray, we need to unpack them
    # Ej = Ej.item(0)
    # j = j.item(0)
    # oldJ, oldEj = oldSMO.selectJ(i, oS, Ei)
    # logging.debug("New Ej: %s, new j: %s; Old Ej: %s, old j: %s", str(Ej), str(j), str(oldEj), str(oldJ))
    # logging.debug("New Ej type: %s, new j type: %s; Old Ej type: %s, old j type: %s", str(type(Ej)), str(type(j)), str(type(oldEj)), str(type(oldJ)))

    ijAreEqualClass = toTheanoBool(T.eq(sS.labels[i], sS.labels[j]))
    L = T.maximum(0,    ifelse(ijAreEqualClass,   sS.alphas[j] + sS.alphas[i] - sS.C,    sS.alphas[j] - sS.alphas[i]))
    H = T.minimum(sS.C, ifelse(ijAreEqualClass,   sS.alphas[j] + sS.alphas[i],           sS.C + sS.alphas[j] - sS.alphas[i]))

    eta = 2.0 * sS.K[i,j] - sS.K[i,i] - sS.K[j,j] #changed for kernel
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
    updatedAlpha = sS.alphas[j] - sS.labels[j]*(Ei - Ej)/eta
    sS.alphas = T.set_subtensor(sS.alphas[j], clipAlpha_(updatedAlpha, H, L), inplace = True)
    updateEk_(sS, j) # add error for alpha[j]
   
    alphaJHasntChanged = toTheanoBool(T.lt(T.abs_(sS.alphas[j] - alphaJold), 0.00001))
    
    updateL = innerL_updateAlphaAndB_(sS, i, Ei, j, Ej, alphaIold, alphaJold)
    earlyret = sS.retlist(0)
    #logging.debug("earlyr: type %s, len %s, vals %s", str(type(updateL)), str(len(updateL)), str(updateL))
    #logging.debug("update: type %s, len %s, vals %s", str(type(earlyret)), str(len(earlyret)), str(earlyret))
    return ifelse(alphaJHasntChanged, earlyret, updateL)

def innerL_updateAlphaAndB_(sS, i, Ei, j, Ej, alphaIold, alphaJold):
    # update alphas[i] by the same amount as alphas[j]
    sS.alphas = T.set_subtensor(sS.alphas[i], sS.alphas[i] + (sS.labels[j]*sS.labels[i]*(alphaJold - sS.alphas[j])), inplace=True)
    # update error for new alpha[i]
    updateEk_(sS, i)
    
    # update b
    b1 = sS.b - Ei- sS.labels[i] * (sS.alphas[i] - alphaIold) * sS.K[i,i] - sS.labels[j] * (sS.alphas[j] - alphaJold) * sS.K[i,j]
    b2 = sS.b - Ej- sS.labels[i] * (sS.alphas[i] - alphaIold) * sS.K[i,j] - sS.labels[j] * (sS.alphas[j] - alphaJold) * sS.K[j,j]
    
    # confitions: use * instead of "and" and + instead of "or"
    alphaInRange = lambda x : toTheanoBool(T.gt(sS.alphas[x], 0) * T.lt(sS.alphas[x], sS.C))
    checkForB2 = ifelse(alphaInRange(j), b2, (b1 + b2)/2.0 )
    sS.b = ifelse(alphaInRange(j), b1, checkForB2)
    
    return sS.retlist(1)

# end of theano innerL implementation



def innerL(i_, oS):
    sS = symbolStruct()
    i = T.iscalar("i")
    
    innerLUpdate = innerL_(sS, i)
    # theano.pp(innerLUpdate[0])
    logging.info("Compiling innerL theano function")
    insyms = sS.symlist() + [i]
    logging.debug("insyms to innerL: %s", str(insyms))
    logging.debug("outsyms of innerLUpdate: %s", str(innerLUpdate))
    compInnerL = theano.function(insyms, innerLUpdate, on_unused_input='ignore')
    logging.info("Compilation finished")
    
    args = sS.arglist(oS) + [i]
    logging.debug("Args to innerL: %s", str(args))
    return sS.saveResults(oS, compInnerL(*args))[0]
    
    

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(dataMatIn,classLabels,C,toler, kTup)
    iteration = 0
    entireSet = True; alphaPairsChanged = 0
    logging.info("Starting main loop")
    while iteration < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                logging.info("fullSet, iteration: %d i:%d, pairs changed %d" % (iteration,i,alphaPairsChanged))
            iteration += 1
        else: #go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                logging.info("non-bound, iteration: %d i:%d, pairs changed %d" % (iteration,i,alphaPairsChanged))
            iteration += 1
        if entireSet:
            entireSet = False #toggle entire set loop
        elif alphaPairsChanged == 0:
            entireSet = True
        logging.info("iteration number: %d" % iteration)
    return oS.b,oS.alphas
