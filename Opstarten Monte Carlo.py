#%% Import
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy.sparse import csr_matrix, lil_matrix
from scipy.io import mmread, mmwrite
from numpy import matrix
#import matplotlib as mpl
#mpl.use("pgf")

#%% Setup
n=37
k1=5 #Size of independent sets
k2=5 #Size of cliques
T=.1
E=0 #A global variable keeping track of the energy

#Used when finding L Ramsey Graphs
L=1
collection=np.array([])

#Used to store a collection of local minima
mincol=np.array([])

# Number cycles in a single round of executing Monte Carlo (before reheating)
r = 10*n*(n-1)

#A global parameter checking whether we go back and forth in the greedy algorithm
forbidden=[]
valleynumber=0
groundnumber=0

#Path Meta Dynamics Stuff:
#Number of times we want to execute Monte Carlo
t=500
# Number of executions in a round in Path Meta Dynamics
s = 100
# A stop value to stop if the energy of a path is 1
stop= False


# %% Definitions

def ones(i):
    ones = np.ones((i,i), dtype=np.int8)
    return ones

#Create Random Graph
def rand_graph():
    graph = np.random.randint(0, 2, (n,n))
    for i in range(n):
        for j in range(i):
            graph[j,i] = graph[i,j]
        graph[i,i]=0
    return graph

# Test if an array is a clique in a certain graph, where the graph is
# represented by its adjacency matrix
def is_clique(matrix, array):
    for i in array:
        for j in array:
            if j!= i:
                if matrix[i,j]==0:
                    return False
    return True

# Finds the number of cliques in a graph (matrix) of size k with dynamic programming.
# attempt is a clique of size l<k, i is an upper bound for vertices
# we already tried to append to our attempt.
# Hence in practice, use i=-1, attempt=[],
# or attempt[a,b] if we want to count the cliques containing the edge (a,b)
def findCliques(matrix, i, attempt, k):
    answer=0
    if len(attempt)==k:
        if is_clique(matrix, attempt):
            answer=1
        else:
            return 0
    for j in range(i+1, len(matrix[0])):
        if j in attempt:
            continue
        else:
            attempt.append(j)
            if is_clique(matrix, attempt):
                if len(attempt)<k:
                    attempt2 = attempt
                    answer += findCliques(matrix, j, attempt2, k)
                else:
                    answer+=1
                attempt.remove(j)
            else:
                attempt.remove(j)
    return answer

#Flip a given edge in a graph
def flip(matrix, a, b):
    matrix[a][b] = 1-matrix[a][b]
    matrix[b][a] = 1-matrix[b][a]
    return matrix

def newflip(graph, a, b):
    if graph.has_edge(a,b)==True:
        graph.remove_edge(a,b)
    else:
        graph.add_edge(a,b)
    return graph

def cgraph(matrix):
    return ones(n) - matrix - np.identity(n, dtype=np.int8)

#The energy function used
def energy(matrix):
    return findCliques(matrix, -1, [], k2) + findCliques(cgraph(matrix), -1, [], k1)

#Calculating the energy part that changes when you flip edge (i,j)
def energy2(matrix, i, j):
    return findCliques(matrix, -1, [i,j], k2) + findCliques(cgraph(matrix), -1, [i,j], k1)

#DeltaH for a graph when flipping the edge (i,j)
def diff(matrix, i, j):
    E1 = energy2(matrix, i, j)
    newgraph = matrix.copy()
    newgraph = flip(newgraph, i, j)
    E2 = energy2(newgraph, i, j)
    return [E2-E1, newgraph]

#Flip a random edge in the graph using the metropolis argument.
def random_flip(matrix):
    global reheating
    global E
    global T
    m = len(matrix[0])
    i = np.random.randint(0, m)
    j = np.random.randint(0, m)
    if i != j:
        difference, newmatrix = diff(matrix, i, j)
        if difference <=0:
            E+=difference
            return newmatrix
        else:
            p = np.random.uniform(0, 1)
            if np.exp(-difference/T)>p:
                E = E+difference
                reheating = False
                return newmatrix
            else:
                return matrix
    else:
        return matrix
    
#Check whether some state is in a valley or saddle point, in the 
#sense that every neighbour has at least the same energy
def local_min(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if diff(matrix, i, j)[0] >= 0:
                continue
            else:
                return False
    return True

#Check whether some state is a local minimum, in the sense that
#every neightbour has a strict higher energy
def strict_local_min(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i!=j:
                if diff(matrix, i, j)[0] > 0:
                    continue
                else:
                    return False
    return True

# Check whether some state is in a flat neighbourhood, i.e.
# every neighbour has the exact same energy
def flat(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if diff(matrix, i, j)[0] == 0:
                continue
            else:
                return False
    return True

#Execute one greedy step on a graph
def greedystep(graph):
    A= 1000
    list=[]
    matrix = nx.to_numpy_array(graph)
    for i in range(n):
        for j in range(n):
            if j != i:
                if diff(matrix, i, j)[0] ==A:
                    list.append([i,j])
                if diff(matrix, i, j)[0]<A:
                    A=diff(matrix, i, j)[0]
                    list=[[i,j]]
    p = np.random.randint(0, len(list))
    a,b=list[p]
    print(a,b)
    graph = newflip(graph, a, b)
    print('Gelukt!')
    print(graph)
    return graph

#Finding a local minimum starting from some state using the
# greedy algorithm
def greedy(matrix):
    matrix2 = matrix.copy()
    global begin
    global forbidden
    global valleynumber
    global groundnumber
    while True:
        print(energy(matrix2))
        A=1
        list=[]
        for i in range(n):
            for j in range(n):
                if j != i:
                    if diff(matrix2, i, j)[0]==A:
                        list.append([i,j])
                    if diff(matrix2, i, j)[0]<A:
                        A=diff(matrix2, i, j)[0]
                        list=[[i,j]]
        if A==1:
            print('local minimum found!')
            return matrix2, True
        else:
            p=np.random.randint(0,len(list))
            a,b = list[p]
            matrix2=flip(matrix2,a,b)
            if A==0 and forbidden[0]==a and forbidden[1]==b:
                print('Going in circles!')
                print(time.time() - begin)
                valleynumber +=1
                return matrix2, False
            forbidden=[a,b]
        if energy(matrix2) ==0:
            groundnumber+=1
            break
    return matrix2, False

#A hashfunction for states, not used
def hashfunc(matrix):
    matrix = tuple(matrix)
    list = ''
    for i in range(n):
        for j in range(i+1, n):
            list = list+str(matrix[i][j])
    answer = int(list, 2)
    return answer % (10**9+7)

#A function to draw the landscape around some graph, does not work
def landscape(graph, t):
    H = nx.from_numpy_array(graph)
    print(H)
    G=nx.Graph()
    G.add_node(H, weight = energy(graph))
    for _ in range(t):
        print('Oude Graaf', H)
        H= greedystep(H)
        print('Nieuwe Graaf', H)
        Hmatrix = nx.to_numpy_array(H)
        print('Oude G', G)
        G.add_node(H, weight = energy(Hmatrix))
        print('Nieuwe G', G)
        for v in G.nodes():
            if np.abs(G.nodes[H]['weight'] - G.nodes[v]['weight']) ==0 and v!=H:
                G.add_edge(v, H)
    return G


#Expanding a Ramsey Graph
def expansion(matrix):
    size = len(matrix[0])
    newmatrix = np.zeros((size+1,size+1), dtype=np.int8)
    for i in range(size):
        for j in range(i):
            newmatrix[i][j] = matrix[i][j]
            newmatrix[j][i] = matrix[j][i]
    newmatrix[size][size] = 1
    return newmatrix

# %% Find Multiple Ramsey Graphs using Monte Carlo Metropolis algorithm
iterationslist = np.array([])
timelist = np.array([])
totreheatlist = np.array([])
for a in range(L):
    T=.1
    adj_matrix = rand_graph()
    E = energy(adj_matrix)
    iterations=0
    totreheat=0
    begin=time.time()
    while E>0:
        reheating = True
        for _ in range(r):
            if E>0:
                adj_matrix=random_flip(adj_matrix)
                iterations+=1
            else:
                break
        if reheating and E>0:
            totreheat +=1
            p = np.random.uniform(1.5, 2.5)
            T=p*T
        else:
            T=0.95*T
    end=time.time()
    totaltime = end-begin
    timelist = np.append(timelist, totaltime)
    iterationslist = np.append(iterationslist, iterations)
    totreheatlist = np.append(totreheatlist, totreheat)
    collection = np.append(collection,adj_matrix)

print('#iterations =', np.mean(iterationslist), 'pm', np.std(iterationslist, ddof=1))
print('runtime =', np.mean(timelist), 'pm', np.std(timelist, ddof=1))
print('#reheatings = ', np.mean(totreheatlist), 'pm', np.std(totreheatlist, ddof=1))

#%% Export the collection of Ramsey Graphs
collection.astype('float32').tofile('examples55_37.dat')

# %% Greedy Algorithm, finding local minima
adj_matrix = rand_graph()
valleynumber=0
groundnumber=0
begin=time.time()
while True:
    adj_matrix = rand_graph()
    matrix, boolean = greedy(adj_matrix)
    if boolean:
        end = time.time()
        print(end-begin)
        mincol = np.append(mincol, matrix)
        break

#%%Exporting the local minima
mincolsave=mincol
mincol.astype('float32').tofile('localmin44_17.dat')

# %%
print(strict_local_min(adj_matrix))

# %%
print(hashfunc(adj_matrix))


# %%
Landscape = landscape(adj_matrix, 10)

# %%
subax1 = plt.subplot(121)
nx.draw(Landscape, with_labels=False, font_weight='bold')
plt.show()
# %%
Test = nx.Graph()
adj_matrix = rand_graph()
graph = nx.from_numpy_array(adj_matrix)
Test.add_node(graph)
subax1 = plt.subplot(121)
nx.draw(Test, with_labels=False, font_weight='bold')
plt.show()
graph = newflip(graph, 0, 1)
graph = newflip(graph, 0, 1)
Test.add_node(graph)
# %% Lowest path from graph to graph

#The hugging of a path, i.e. changing a path from ..., e1, e2, ...
#to ..., e, e1, e2, e, ...
#Instead of the graph sequence G1, G, G2 we now get from G1 to G2
#via G1, H1, G', H2, G2. So we walk around G.
#Hugging only works if we are not at the end of our path, and
#our new edge e must not be equal to one of its neighbours.
def knuffelen(matrix, path, elist):
    newmatrix = matrix.copy()
    newpath=path.copy()
    newelist=elist.copy()
    eindex = np.argmax(newelist)
    if eindex == 0 or eindex==len(newelist)-1:
        return newpath, newelist
    list = np.random.choice(n, 2, replace=False)
    list.sort()
    b,a = list
    print('(a,b) = ', a, b)
    if a == newpath[eindex][0] and b == newpath[eindex][1]:
        return newpath,newelist
    if a == newpath[eindex-1][0] and b == newpath[eindex-1][1]:
        return newpath,newelist
    if eindex != len(newelist)-2 and a == newpath[eindex+1][0] and b == newpath[eindex+1][1]:
        return newpath,newelist
    if eindex !=1 and a == newpath[eindex-2][0] and b == newpath[eindex-2][1]:
        return newpath,newelist
    newpath = np.insert(newpath, eindex, [a,b], axis=0)
    newpath = np.insert(newpath, eindex-1, [a,b], axis=0)
#Updating the energy list:
    for i in range(eindex):
        a,b = newpath[i]
        newmatrix = flip(newmatrix, a, b)
    currenergy = energy(newmatrix)
    newelist = np.insert(newelist,eindex ,currenergy)
    a,b = newpath[eindex]
    info = diff(newmatrix, a, b)
    currenergy += info[0]
    newmatrix = info[1]
    newelist[eindex+1] = currenergy
    a,b = newpath[eindex]
    info = diff(newmatrix, a, b)
    currenergy+=info[0]
    newmatrix = info[1]
    newelist = np.insert(newelist, eindex+2, currenergy)
    return newpath, newelist

#The flipping of a step in our path, i.e. changing ..., e1, e2, ...
#to ..., e2, e1, ...
#Instead of the graph sequence G1, G, G2 we now get from G1 to
#G2 via G1, G', G2.
#By flipping, we can get two of the same edges next to each other,
#meaning we have changed ..., e2, e1, e2, ... to ..., e2, e2, e1,...
#In this case, we need to delete the steps e2, e2 as they don't contribute.
def pathflip(matrix, path, elist):
    newmatrix = matrix.copy()
    newelist=elist.copy()
    newpath = path.copy()
    eindex = np.argmax(newelist)
    print('eindex =', eindex)
    if eindex == 0:
        return newpath, newelist
    if eindex == len(newelist)-1:
        return newpath, newelist
    else:
        dummy = newpath[eindex].copy()
        newpath[eindex] = newpath[eindex-1]
        newpath[eindex-1] = dummy
        for i in range(eindex):
            a,b = newpath[i]
            newmatrix = flip(newmatrix, a, b)
        newelist[eindex] = energy(newmatrix)
#Deleting a `palindrome' in our path, so deleting steps like
#e, e or e1, e2, e2, e1 if we have created this by flipping
        if len(newpath)> eindex-1:
            for i in range(eindex):
                if newpath[eindex-i][0] == newpath[eindex+1-i][0] and newpath[eindex-i][1] == newpath[eindex+1-i][1] and eindex>=i:
                    newpath = np.delete(newpath, eindex+1-i, axis=0)
                    newpath=np.delete(newpath, eindex-i, axis=0)
                    newelist = np.delete(newelist, eindex+1-i, axis=0)
                    newelist=np.delete(newelist, eindex-i, axis=0)
            if i<2:
                for j in range(eindex):
                    if newpath[eindex-2-j][0] == newpath[eindex-1-j][0] and newpath[eindex-1-j][1] == newpath[eindex-j][0] and eindex-2>=j:
                        newpath = np.delete(newpath, eindex-1-j, axis=0)
                        newpath = np.delete(newpath, eindex-2-j, axis=0)
                        newelist = np.delete(newelist, eindex-1-j, axis=0)
                        newelist = np.delete(newelist, eindex-2-j, axis=0)
                    else:
                        break
        return newpath, newelist

#Finding the lowest path between two graphs matrix1, matrix2 by
#using a Monte Carlo approach. A small step we can do is either
#hugging or flipping around the graph in our path with the highest
#energy.
def initialpath(matrix1, matrix2):
    matrix = (matrix1 + matrix2)%2
    currmatrix = matrix1
    currenergy = energy(matrix1)
    elist = np.array([currenergy])
    path = np.argwhere(matrix)
    path = path[(path[:,:1]-path[:,1:]).flatten()>0]
    for i in range(len(path)):
        a,b = path[i]
        info = diff(currmatrix,a,b)
        currenergy += info[0]
        elist = np.append(elist, currenergy)
        currmatrix = info[1]
    return path, elist

def lowestpathstep(matrix1, path, elist):
    global reheating
    global stop
    p = np.random.uniform(0, 1)
    if p<0.25: #Execute hugging
        print('Hugging!')
        newpath, newelist = knuffelen(matrix1, path, elist)
    else: #Execute flipping
        print('Flipping!')
        newpath, newelist = pathflip(matrix1, path, elist)
    eindex = np.argmax(elist)
    pathenergy = elist[eindex]
    print('elist=',elist)
    print('newelist=', newelist)
    neweindex = np.argmax(newelist)
    newpathenergy=newelist[neweindex]
    print('index, newenergy =',neweindex, newpathenergy)
    if newpathenergy ==1:
        print('newpathenergy = ', newpathenergy)
        stop=True
        return newpath, newelist
    if newpathenergy<=pathenergy:
        print('Yes')
        path = newpath
        elist=newelist
    else:
        q = np.random.uniform(0, 1)
        print('checking', newpathenergy, pathenergy)
        if q< np.exp(-(newpathenergy-pathenergy)/T):
            print('Accepted')
            reheating = False
            path = newpath
            elist = newelist
    return path, elist

# %%
p1 = mmread("C:/Users/kevin/OneDrive/Documenten/Grafencollectie/r44_16_1.mtx")
p2 = mmread("C:/Users/kevin/OneDrive/Documenten/Grafencollectie/r44_16_2.mtx")
p1 = np.array(np.matrix(np.array(p1)))[0,0]
p1 = p1.toarray()
p2 = np.array(np.matrix(np.array(p2)))[0,0]
p2 = p2.toarray()
path, elist = initialpath(p1, p2)
T=.1
begin=time.time()
for _ in range(t):
    reheating= True
    for __ in range(s):
        path, elist = lowestpathstep(p1, path, elist)
        if stop:
            break
    if stop:
        break
    if reheating:
        T = 1.7*T
    else:
        T=0.95*T
end=time.time()
print(end-begin)

# %% Finding Ramsey graph by expanding existing Ramsey graphs.
new = expansion(adj_matrix)
for a in range(L):
    adj_matrix = new
    n +=1
    E = energy(adj_matrix)
    iterations=0
    while E>0:
        reheating = True
        for _ in range(r):
            if E>0:
                adj_matrix=random_flip(adj_matrix)
                iterations+=1
            else:
                break
        if reheating:
            p = np.random.uniform(1.5, 2.5)
            T=p*T
        else:
            T=0.95*T
    collection = np.append(collection,adj_matrix)
# %%
p = mmread("C:/Users/kevin/OneDrive/Documenten/localmin44_17eigs.mtx")
plt.figure()
x=2*np.ones(len(p[1]))
print(p[0])
for i in range(len(p)):
    plt.plot(p[i], i*np.ones(len(p[0])), '.')
plt.xlim(-5, 9)
plt.ylim(-1, 9)
plt.savefig('localminr44_17eigs1.pgf')
# %%
p1 = mmread("C:/Users/kevin/OneDrive/Documenten/Grafencollectie/localminr44_17.mtx")
#p1 = np.array(np.matrix(np.array(p1)))[0,0]
#p1 = p1.toarray()
graph = nx.from_numpy_array(p1)
nx.draw(graph)
plt.show()
nx.write_graph6(graph, 'anedge.g6')
# %%


file_path = 'examples55_37.dat'
 
with open(file_path, 'r') as file:
    for line in file:
        print(line.strip())

# %% Plotting the found transition path on R(3,4) on 8 vertices

ylijst = elist
xlijst = np.linspace(0, 1, num=len(elist))
pad, iniylijst = initialpath(p1, p2)
inixlijst = np.linspace(0, 1, num=len(iniylijst))

plt.figure()
plt.plot(xlijst, ylijst, 'r')
plt.plot(inixlijst, iniylijst, 'green')
plt.xlabel("Graphs")
plt.ylabel("Energy")
plt.xlim((0, 1))
plt.ylim((0,20))
plt.xticks([])
plt.minorticks_on()
plt.savefig('tps44_16.pdf')
plt.show()


 # %%
