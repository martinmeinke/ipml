from numpy import *
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse
import logging
import smo as oldSMO


def colToRow(x):
    # dimshuffle converts the column vector into a row vector
    return x.dimshuffle('x', 0)

def selectRandomJ_(i, m):
    rstr = RandomStreams()
    # TODO: make somehow sure that the random integer is not i
    return rstr.random_integers(None, 0, m, ndim=0)

def clipAlpha(aj,H,L):
    # in theano: aj = ifelse(T.gt(aj, H), H, aj); aj = ifelse(T.gt(L, aj), L, aj);
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
        self.m = self.K.shape[0]
        self.alphas = T.col()
        self.b = T.scalar()
        self.eCache = T.matrix()

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
    deltaErrors = theano.scan(lambda k: abs(Ei - calcEk_(sS.alphas, sS.labels, sS.K, sS.b, k)), sequences=[validEcacheList])[0]
    selectMaxError = T.max_and_argmax(deltaErrors)
    
    # if we donÄt have cached errors, yet, we need code to select a random j and Ej
    randomJ = selectRandomJ_(i, sS.m)
    randomJAndError = [calcEk_(sS.alphas, sS.labels, sS.K, sS.b, randomJ), randomJ]

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

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i]*Ei >= -oS.tol or oS.alphas[i] >= oS.C) and \
       (oS.labelMat[i]*Ei <=  oS.tol or oS.alphas[i] < 0):
        return 0

    Ej, j = selectJ(i, oS, Ei) # both return values are ndarrray, we need to unpack them
    Ej = Ej.item(0)
    j = j.item(0)
    # oldJ, oldEj = oldSMO.selectJ(i, oS, Ei)
    # logging.debug("New Ej: %s, new j: %s; Old Ej: %s, old j: %s", str(Ej), str(j), str(oldEj), str(oldJ))
    # logging.debug("New Ej type: %s, new j type: %s; Old Ej type: %s, old j type: %s", str(type(Ej)), str(type(j)), str(type(oldEj)), str(type(oldJ)))
    
    
    alphaIold = oS.alphas[i].copy()
    alphaJold = oS.alphas[j].copy()

    if oS.labelMat[i] != oS.labelMat[j]:
        L = max(0, oS.alphas[j] - oS.alphas[i])
        H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
    else:
        L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
        H = min(oS.C, oS.alphas[j] + oS.alphas[i])
    if L==H:
        logging.info("L==H")
        return 0

    eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
    if eta >= 0:
        logging.info("eta>=0")
        return 0

    update =  oS.alphas[j] - oS.labelMat[j]*(Ei - Ej)/eta
    logging.debug("j is %s, Ej is %s, alphas[j] shape: %s, update shape: %s", str(j), str(Ej), str(np.shape(oS.alphas[j])), str(np.shape(update)))
    oS.alphas[j] = update
    oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
    updateEk(oS, j) #added this for the Ecache
    if abs(oS.alphas[j] - alphaJold) < 0.00001:
        logging.info("j not moving enough")
        return 0

    oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
    updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
    b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
    b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
    if 0 < oS.alphas[i] and oS.C > oS.alphas[i]:
        oS.b = b1
    elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]:
        oS.b = b2
    else:
        oS.b = (b1 + b2)/2.0
    # TODO: find a solution for that workaround so b is never a 1x1 matrix
    oS.b = oS.b.item(0)
    return 1

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
