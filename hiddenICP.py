import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from collections import Counter

def funtionICP(X,Y,ExpInd,alpha=0.1,mode="asymptotic",intercept=False):
    if isinstance(X, list) and X.isnumeric():
        X = np.asmatrix(X,ncol=1)
    if not isinstance(X, np.ndarray)  and not isinstance(X, pd.DataFrame):
        raise ValueError("'X' must be a matrix or data frame")
    if not isinstance(Y, np.ndarray):
        raise ValueError("'Y' must be a vector")
    if X.shape[0] <= X.shape[1]:
        raise ValueError("hiddenICP not suitable for high-dimensional data (at the moment) \n -- need row > column but have nrow(X)= {} and ncol(X)={}".format(X.shape[0], X.shape[1]))
    if not isinstance(ExpInd, list):# If ExpInd is not a list
        if len(ExpInd) != len(Y):
            raise Exception("if `ExpInd' is a vector, it needs to have the same length as `Y'")
        uni= np.unique(ExpInd)
        if len(uni) == 1:
            raise Exception("There is just one environment ('ExpInd'= {} for all observations) and the method needs at least two distinct environments sep = ".format(uni[1]))
        if min(Counter(ExpInd)) <= 2:
            print("\nOut put of 'table(ExpInd)':\n ")
            print(Counter(ExpInd))
            raise Exception("one environment has just one or two observations (as supllied by 'ExpInd'); there need to be at least 3 (and ideally dozens) of observations in each environment; the out put of 'table(ExpInd)' is given below to show the number of observations in each unique environment as supplied by 'ExpInd'")
        K= len(uni)
        ExpIndNEW = list()
        for uc in range(0,K):
            ExpIndNEW[uc] = np.where(ExpInd == uni[uc])          
            setattr(ExpIndNEW[uc],"value",uni[uc])
        ExpInd = ExpIndNEW # Now ExpInd is a list
        del ExpIndNEW
    else : #if ExpInd is a list
        if min(ExpInd) < 1 :
            raise Exception("if `ExpInd' is a list with indicies of observations, \n minimal entry has to be at least 1 but is {}".format(min(ExpInd)))
        if max(ExpInd) > len(Y):
            raise Exception("if `ExpInd' is a list with indicies of observations, \n maximal entry has to be at most equal \n to the length {} of the observations but is {}".format(len(Y), max(ExpInd)))
    X = pd.DataFrame(X)
    '''if len(ucol = set(X.shape[1])) < min(3, X.shape[1]) :
       colnames(X) = paste("Variable",1:X.shape[1],sep="_")'''
    colX = X.columns
    if intercept:
        X = np.column_stack((np.repeat(1,X.shape[0]), X))
    K = len(ExpInd)
    p = X.shape[1]
    n = X.shape[0]

    kc = 1
    if K > 2 :
        KC = K 
    else:
        KC = 1
        
    ConfInt = np.zeros(0, shape = [2,p])
    pvalues = np.repeat(1,p)
    for kc in range(0, KC):
        ins = ExpInd[kc]
        out = (1,n)[-ins]
        DS = (np.transpose(X[ins:]).dot(X[ins:]))/len(ins) - (np.transpose(X[out:]).dot(X[out:]))/len(out)
        Drho = (np.transpose(X[ins:]).dot(Y[ins]))/len(ins) - (np.transpose(X[out:]).dot(Y[out]))/len(out)
        DSI = np.linalg.solve(DS, Drho)
    
        
        betahat = pd.to_numeric(np.linalg.solve( DS,Drho))
        if kc == 1:
            betahatall = betahat 
        else:
            betahatall = betahatall + betahat
        Zin = np.zeros(shape=[len(ins), p])
        Zout = np.zeros(shape=[len(out),p])
        for i in range(0,len(ins)):
            tmp = DSI * X[ins[i],]
            Zin[i,] = pd.to_numeric(- tmp * sum(tmp*Drho) + Y[ins[i]] *tmp)
        
        for i in range(0, len(out)):
            tmp = DSI * X[out[i],]
            Zout[i,] = pd.to_numeric(- tmp * sum(tmp*Drho) + Y[out[i]] *tmp)
        
        sigmavec = math.sqrt(np.diag((np.cov(Zin)/len(ins)+np.cov(Zout)/len(out))))

        pvalues = min(pvalues, 2*K* (1-norm.cdf( abs(betahat)/max(pow(10,-10),sigmavec),df=n-1)))
        
        addvar = norm.ppf(max(0.5,1-alpha/(2*K))) * sigmavec
        maximineffectsN = np.sign(betahat) * max( 0, abs(betahat) - addvar)
        ConfInt[1:] = max(ConfInt[1:], betahat - addvar, True) 
        ConfInt[2:] = min(ConfInt[2:], betahat + addvar, True) 
        if kc==1 :
            maximineffects = maximineffectsN
        else:
            for varc in range(1,p+1):
                if abs(maximineffectsN[varc]) > abs(maximineffects[varc]) :
                    maximineffects[varc] = maximineffectsN[varc]
            
    
    betahat = betahatall/KC
    maximinCoefficients = maximineffects
    if intercept:
        betahat = betahat[-1]
        maximinCoefficients = maximinCoefficients[-1]
        ConfInt = ConfInt[:-1]
        pvalues = pvalues[-1]
    
    ConfInt = pd.apply(ConfInt,2,result_type = 'sort',decreasing=True)
    retobj = list(betahat=betahat,maximinCoefficients=maximinCoefficients,ConfInt=ConfInt,pvalues=pvalues,colnames=colX,alpha=alpha)
    #class(retobj) <- "hiddenInvariantCausalPrediction"
    return retobj 