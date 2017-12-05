
# coding: utf-8
import sparsifyDensify
import pandas as pd
import numpy as np
import copy
from operator import itemgetter, attrgetter
from itertools import compress
import junctionGraph

def iscycle(korif, akmi):
    #Test whether there will be a circle if a new edge is added
    """
    Test whether there will be a circle if a new edge is added
    korif is set of vertices in the graph
    akmi is edge we insert in graph
    c = 1 if we have circle, else c = 0
    """
    g=np.max(korif)+1
    #print("g",g)
    #print("korif",korif)
    c=0
    n=len(korif)
    #print("a1",korif[0][akmi[0]])
    #print("a2",korif[0][akmi[1]])
    if korif[0][akmi[0]]==0 and korif[0][akmi[1]]==0:
        korif[0][akmi[0]]=g
        korif[0][akmi[1]]=g
        #print("korif ==0 ==0",korif)
    elif korif[0][akmi[0]]==0:
        korif[0][akmi[0]]=korif[0][akmi[1]]
    elif korif[0][akmi[1]]==0:
        korif[0][akmi[1]]=korif[0][akmi[0]]
    elif korif[0][akmi[0]]==korif[0][akmi[1]]:
        c=1
    else:
        m=np.max(korif[0][akmi[0]],korif[0][akmi[1]])
        for i in range(n):
            if korif[0][i]==m:
                korif[0][i]=np.min(korif[0][akmi[0]],korif[0][akmi[1]])
                                   
    return korif,c


# In[177]:

def fysalida(A, col):
    #swap matrix's rows, because we sort column (col) by descending order
    #col is column we want to sort
    _, c= A.shape

    if col < 1 or col > c or np.fix(col) != col:
        raise ValueError("second argumment takes only integer values between 1 and col")
    A=sorted(A, key=itemgetter(col), reverse=True)
    A=np.asarray(A)
    return A    


# In[178]:

def kruskal(PV, numV):

    """
    Kruskal algorithm for finding maximum spanning tree
    @params:
    PV is nx3 martix. 1st and 2nd number's define the edge (2 vertices) and the 3rd is the edge's weight.
    numV is number of vertices
    @returns:
    Et is adjacency matrix of maximum spanning tree
    w is maximum spanning tree's weight
    """

    Et = np.zeros((numV, numV), dtype=bool);
    if PV.shape[0] == 0:
        W = 0;
        return Et,W
    num_edge = PV.shape[0] 
    #sort PV by descending weights order.
    PV = fysalida(PV,2)
    korif = np.zeros((1, numV),dtype=np.int64)
    insert_vec =np.ones((num_edge, 1),dtype=bool)
    for i in range( num_edge):
        akmi = PV[i][0:2]
        korif, c= iscycle(korif, akmi)
 
        #insert the edge iff it does not introduce a circle
        insert_vec[i] = (c == 0)
        #Create maximum spanning tree's adjacency matrix
        if insert_vec[i]:
            #print("a1",PV[i][0])
            #print("a2",PV[i][1])
            Et[PV[i][0], PV[i][1]] = True
            Et[PV[i][1], PV[i][0]] = True
    #Calculate maximum spanning tree's weight
    #print("PV",PV)
    #print("insert_vec",insert_vec)
    W = np.sum(np.asarray(list(compress(PV, insert_vec)))[:,2])
    
    return Et,W


# In[179]:

#Adopted from https://github.com/kylemin/HEX-graph/blob/master/%2BhexGraph/junctionTree.m
def junctionTree(cliques, numV):
    numC = len(cliques)
    edgesJG =np.zeros((0, 3),dtype=np.int64)
    for c1 in range(numC - 1):
        for c2 in range(c1 + 1,numC):
            #the weight of each edge is the variable number after intersection
        
        
            weight=len(set(cliques[c1]).intersection(cliques[c2]))
            #print("cIntersect",cIntersect)
            if weight > 0:#[np.concatenate((tuple(variables[v])[0], [c]), axis=0)]
                #print("weight",weight)
                row=np.asarray([c1, c2, weight])
                edgesJG = np.vstack((edgesJG, row))
    #print("edgesJG",edgesJG)
    Ej, Wj = kruskal(edgesJG, numV)
    #cliqParents records each clique's parent clique index (-1 for root)
    cliqParents = np.zeros((numC, 1),dtype=np.int64)
    cliqParents= np.asarray([x - 1 for x in cliqParents])
    #childVariables records eqch clique's children clique indices (empty  for leaf)
    childVariables=np.empty((numC,1),dtype=object)
    #the clique indices in up passage pass sequence.
    upPass = []
    #depth-first search
    #Select an arbitrary clique as root.
    rootJT = 0
    visited =np.zeros((numC, 1),dtype=bool)
    cliq = rootJT

    while True:
        #print("Ej",Ej)
        #print("cliq",cliq)
        #print("Ej[:, cliq]",Ej[:, cliq])
        cliqNei =np.where(Ej[:, cliq])[0]
        #print("cliqNei",cliqNei)
        #Visit all current clique's children who has not been visited yet.
        #If no children (leaf) or all children have been visited, then visit current clique and back to parent.
        #message and to back to parents. If no parent (root), then stop
        visitChild = False
        cParent = cliqParents[cliq]
        for n in range(len(cliqNei)):
            cChild = cliqNei[n]
            #print("cChild",cChild)
            if (cChild != cParent) and (visited[cChild]==False):
                visitChild = True
                break
        if visitChild:
            cliqParents[cChild] = cliq
            cliq = int(cChild)
        else:
            #visit current clique
            visited[cliq] = True
            c_adjacency = Ej[:, cliq]
            if cParent >= 0:
                c_adjacency[cParent] = False
            #print("c_adjacency",c_adjacency)
            #print("np.where(c_adjacency)",np.where(c_adjacency))
            temp=[np.where(c_adjacency)[0]]
            #print("temp",temp)
            if len(temp)!=0:
                #print("childVariables before",childVariables)
                #print("childVariables[cliq] before",childVariables[cliq])
                #print("cliq",type(cliq))
                childVariables[cliq] = [np.where(c_adjacency)[0]]
            #print("childVariables",childVariables)
            #print("upPass",upPass)
            #print("cliq 2",cliq)
            if len(upPass)==0:
                upPass=[cliq]
                #print("upPass cliq",type(upPass))
            else:
                #print("upPass in",upPass)
            
                upPass.append(cliq)
            #print("upPass after",upPass)
            #print("cliq",cliq)
            #print("cliqParents",cliqParents)
            if cliqParents[cliq] >= 0:
                cliq = int(cliqParents[cliq])
            else:
                break
    assert(len(upPass) == numC) 
    return cliqParents,childVariables,upPass


