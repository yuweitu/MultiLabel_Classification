
# coding: utf-8
#This function is called after sparsifyDensify.py
import sparsifyDensify
import pandas as pd
import numpy as np
import copy

#junction graph
#Adopted from https://github.com/kylemin/HEX-graph/blob/master/%2BhexGraph/junctionGraph.m
#Helpful references: 1. http://www.inf.ed.ac.uk/teaching/courses/pmr/slides/jta-2x2.pdf
## 2. http://ai.stanford.edu/~paskin/gm-short-course/lec3.pdf
def junctionGraph(Ehs, Ees):
    numV=Ehs.shape[0]
    #Moralization
    Eelim=Ehs+Ehs.T+Ees

    #Generate a variable elimination sequence
    unvisited =np.ones((numV, 1), dtype=bool)
    elimOrder =np.zeros((numV, 1), dtype=np.int)
    Etemp = copy.copy(Eelim)
    k = 0
    while k!=numV:
        numNei = np.sum(Etemp, 1)[np.newaxis].T #to make numpy array a column vector
    
        minNode=min(numNei[unvisited])
   
        for i in range(numV):
            #Find minimal fill
            if unvisited[i] and numNei[i]==minNode:
                for j in range(numV):
                    if Etemp[i,j]:
                        Etemp[i, j] =0
                        Etemp[j, i] =0
                unvisited[i] = False
                k = numV - np.sum(unvisited)
                elimOrder[k-1]=i
    #eliminate nodes to get elimination cliques
    cliques=np.empty((numV,1),dtype=object)
    widthJT = 0
    for vid in range(numV):
    
        v = int(elimOrder[vid])
        #Find its neighbours and form a clique.

        vNei=np.where(Eelim[:, v])[0]
    
        numVN=len(vNei)
        assert((numVN)>=1 or (vid==(numV-1)))
        cliques[v]=[np.append([v],[vNei])]
   
        widthJT = max(widthJT, numVN)
        for n1 in range(numVN):
            for n2 in range(n1+1,numVN):
                Eelim[vNei[n1], vNei[n2]] = 1
                Eelim[vNei[n2], vNei[n1]] = 1
        Eelim[:, v] = 0
        Eelim[v, :] = 0
    #Find maximal cliques from all elimination cliques
    numC = len(cliques)
    keep=np.ones((numC, 1), dtype=bool)
    for c1 in range(numC):
        for c2 in range(c1+1,numC):
            if keep[c1]!=True or keep[c2]!=True:
                continue
            #take intersection of two cliques 
            if type(tuple(cliques[c1])[0])==int:
                len_cliques_c1=1
                if type(tuple(cliques[c2])[0])==int:
                    len_cliques_c2=1
                    if tuple(cliques[c1])[0]==tuple(cliques[c2])[0]:
                        cIntersect=tuple(cliques[c1])[0]
                    else:
                   
                        cIntersect=np.empty( shape=(0, 0) )
                else:
                    #cliques[c2] is ndarray
                    len_cliques_c2=len(tuple(cliques[c2])[0])
                    if tuple(cliques[c1])[0] in tuple(cliques[c2])[0]:
                        cIntersect=tuple(cliques[c1])[0]
            else:
                #cliques[c1] is ndarray
                len_cliques_c1=len(tuple(cliques[c1])[0])
                if type(tuple(cliques[c2])[0])==int:
                    len_cliques_c2=1
                    if tuple(cliques[c2])[0] in tuple(cliques[c1])[0]:
                        cIntersect=tuple(cliques[c2])[0]
                    else:
                        cIntersect=np.empty( shape=(0, 0) )
                else:
                    len_cliques_c2=len(tuple(cliques[c2])[0])
                    cIntersect=list(set(tuple(cliques[c1])[0]).intersection(tuple(cliques[c2])[0]))
       
            #if one clique contains another, then remove the small one
            if len(cIntersect) == len_cliques_c1:
                keep[c1] =False
            elif len(cIntersect) == len_cliques_c2:
                keep[c2] = False

    cliques=cliques[keep]

    #Record which cliques each variable appears in, so that it can be marginalized efficiently.
    numC = len(cliques)
    variables=np.empty((numV,1),dtype=object)
    for c in range(numC):
        #print("cliques[c])",tuple(cliques[c]))
        vc = tuple(cliques[c])
    
        for vid in range(len(vc)):
            v = vc[vid]
            if np.any(tuple(variables[v])[0])!=None:  
                variables[v] =[np.concatenate((tuple(variables[v])[0], [c]), axis=0)]
            else:
            
                variables[v] =[np.asarray([c])]
    #record how many times a variable appears
    numVar=np.zeros((numV, 1))
    for v in range(numV):
        numVar[v] = len(tuple(variables[v])[0]);
    return cliques,variables,numVar


