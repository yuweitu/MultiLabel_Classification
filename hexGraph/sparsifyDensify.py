
# coding: utf-8

# In[13]:

#Adopted from https://github.com/kylemin/HEX-graph/blob/master/%2BhexGraph/sparsifyDensify.m


# In[1]:

import pandas as pd
import numpy as np
import copy

def sparsifyDensify(Eh, Ee):
    Ehs = copy.copy(Eh) #shallow copy
    Ehd = copy.copy(Eh)
    Ees = copy.copy(Ee)
    Eed = copy.copy(Ee)
    numV = Eh.shape[0]
    #For Eh
    for i in range(numV):
        
        qT = [] #Temporary queue
        
        qD = []
        for j in range(numV):
            
            #Find every descendent
            if Eh[i, j]==1:
                qT.append(j)
            
        #Find if there is a possible directed path
        while len(qT)>0:
            t=qT[0]
            qT.pop(0)
            for j in range(numV):
                if Eh[t, j]==1:
                    qT.append(j)
                    qD.append(j)
        for j in range(len(qD)):
            s = qD[j]
            if Eh[i, s]:
                #Sparsify
                Ehs[i, s] = 0
            else:
                #Densify
                Ehd[i, s] = 1
    #About Ee (check exclusion edges) 
    for i in range(numV-1):# This is because Ee is symmetric matrix
        for j in range(i+1,numV):
            qi = []  # Temporary queue
            qAi = [] # Queue for ancestors of node i
            qj = []  # Temporary queue
            qAj = [] # Queue for ancestors of node j 
            if Eh[i, j]==0 and Eh[j, i]==0:
                qi.append(i)
                qAi.append(i)
                while len(qi)>0:
                    s=qi[0]
                    qi.pop(0)
                    for k in range(numV):
                        if Eh[k, s]:
                            qi.append(k)
                            qAi.append(k)
                qj.append(j)
                qAj.append(j)
                while len(qj)>0:
                    s=qj[0]
                    qj.pop(0)
                    for k in range(numV):
                        if Eh[k, s]:
                            qj.append(k)
                            qAj.append(k)
            exist = False
            #Find if there is any e between nodes of qAi and qAj
            for m in range(len(qAi)):
                s = qAi[m]
                for n in range(len(qAj)):
                    t = qAj[n]
                    if Ee[s, t]:
                        if (s==i and t==j)==False and (s==j and t==i)==False:
                            exist = True
                            break
                if exist:
                    break
            if exist:
                if Ee[i, j]:
                    #Sparsify
                    Ees[i, j] = 0
                    Ees[j, i] = 0   
                else:
                    #Densify
                    Eed[i, j] =1
                    Eed[j, i] =1
    return Ehs, Ees, Ehd, Eed




