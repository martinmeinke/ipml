from numpy import *
import logging
from theanoSMO import fkKernelTrans

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    logging.info("Taking random j: %d", j)
    return j
    # return 456

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def kernelTrans(X, A, kernel): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kernel[0]=='lin': K = X * A.T   #linear kernel
    elif kernel[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kernel[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
        # Use theano implementation for this
        self.K = mat(fkKernelTrans(self.X, kTup))
        #self.K = mat(zeros((self.m,self.m)))
        #logging.info("Applying kernel transformation to data")
        #for i in range(self.m):
        #    self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    # logging.debug("%s nonzero values in eCache", str(len(nonzero(oS.eCache[:,0].A)[0])))
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            logging.debug("Delta of error %d is %f", k, deltaE)
            if (deltaE > maxDeltaE):
                logging.debug("Using k=%d because %f > %f", k, deltaE, maxDeltaE)
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    logging.debug("Update error %d to %f", k, Ek)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i]*Ei >= -oS.tol or oS.alphas[i] >= oS.C) and \
       (oS.labelMat[i]*Ei <=  oS.tol or oS.alphas[i] < 0):
        return 0

    j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
    alphaIold = oS.alphas[i].copy()
    alphaJold = oS.alphas[j].copy()

    if oS.labelMat[i] != oS.labelMat[j]:
        L = max(0, oS.alphas[j] - oS.alphas[i])
        H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
    else:
        L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
        H = min(oS.C, oS.alphas[j] + oS.alphas[i])
    if L==H:
        logging.debug("L==H")
        return 0

    eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
    if eta >= 0:
        logging.debug("eta>=0")
        return 0

    logging.debug("Ei= %f, Ej= %f, eta= %f", Ei, Ej, eta)
    logging.debug("H= %f, L= %f", H, L)
    updatedAlpha = oS.alphas[j] - (oS.labelMat[j] * (Ei - Ej) /eta)
    updatedAlpha = clipAlpha(updatedAlpha, H, L)
    logging.debug("Updated alpha %d is %f", j, updatedAlpha)
    oS.alphas[j] = updatedAlpha
    updateEk(oS, j) #added this for the Ecache
    if abs(oS.alphas[j] - alphaJold) < 0.00001:
        logging.debug("j not moving enough")
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
    logging.debug("b= %f", oS.b)
    return 1

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(dataMatIn,classLabels,C,toler, kTup)
    iteration = 0
    entireSet = True; alphaPairsChanged = 0
    logging.info("Starting main loop")
    entireSetRange = range(oS.m)
    while iteration < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        bounds = entireSetRange if entireSet else nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
        setName = "fullSet" if entireSet else "non-bound"
        runI = 0
        logging.info("This iteration will consist of %d steps", len(bounds))
        for i in bounds:
            alphaPairsChanged += innerL(i,oS)
            if runI > 0 and runI % 100 == 0:
                logging.info("%s, iteration: %d i:%d, pairs changed %d" % (setName, iteration,i,alphaPairsChanged))
            runI += 1
        iteration += 1

        if entireSet:
            entireSet = False #toggle entire set loop
        elif alphaPairsChanged == 0:
            entireSet = True
        logging.info("iteration number: %d" % iteration)
    return oS.b,oS.alphas
