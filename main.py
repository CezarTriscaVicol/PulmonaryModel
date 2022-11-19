from asyncio import new_event_loop
from concurrent.futures import thread
from gettext import find
from multiprocessing import current_process
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import statistics
import operator
import scipy as sp

xp = np # or cp
spx = sp # or cpx.scipy

nodeCount = 1024
leafCount = math.ceil(nodeCount/2)
firstLeaf = nodeCount - leafCount

readFromFile = True
saveToFile = False

if(not readFromFile):
    initialPressureValues = xp.full(nodeCount-1, 1033)
    initialFlowValues = xp.full(nodeCount-1, 0)
    initialVolumeValues = xp.full(leafCount, (2.5)/leafCount)
    initialApl = -15
    initialRpl = 5  
else:
    input = xp.load('initialInputs/initialInputs'+str(nodeCount)+'.npy')
    initialPressureValues = input[2:nodeCount+1]
    initialFlowValues = input[nodeCount+1:2*nodeCount]
    initialVolumeValues = input[2*nodeCount:]
    initialApl = input[0]
    initialRpl = input[1]  

Pat = 1033                 # cmH2O       Atmospheric pressure
correctionConstant = 1.85
miu = 1.81 * 1e-5          # kg/(m*s) - Viscosity of air
ro = 1.204                 # kg/m^3   - Density of air
eps = 1e-6                
FRC = 2.5                  # L - Functional Residual Capacity
Vmin = 2.5                 # L - Volume remaining after exhalation
Vmax = 2.9                 # L - Volume at the end of inhalation 
Vdepth = Vmax - Vmin       # L - Volume inhaled / exhaled
Apl = initialApl    # cmH2O - Mean pleural pressure
Rpl = initialRpl     # cmH2O - Pleural pressure range
fz  = 1      # Linear gradient factor
breathLength = 4       #s length of a normal breathing cycle
maxDiameter = 0.01     # m maximum diameter of a branch
RdS = 1.4              # anti-log of the slope of the airway diameter plotted against Strahler order
Compliance = (2.5/(nodeCount/2))/(Pat + 15) # L/cmh2o  Compliance of the acinar Region

G = nx.Graph()
nodeList = xp.arange(1, nodeCount)              # List of all nodes/branches without root
terminalList = xp.arange(firstLeaf, nodeCount)  # List of all terminal nodes/branches
nonTerminalList = xp.arange(1, firstLeaf)       # List of all non-terminal nodes/branches (without root)
fatherList = xp.concatenate((xp.array([0]), xp.repeat(xp.arange(1, firstLeaf), 2))) # List of all fathers
leftSonList = xp.add(nonTerminalList, nonTerminalList)   # List of all left sons
rightSonList = xp.add(leftSonList, 1)                    # List of all right sons 
oneEachDepth = []
val = 1
while(val < nodeCount):
    oneEachDepth.append(val)
    val = val * 2
G.add_node(0, P = Pat)
for i in nonTerminalList: G.add_node(i, P = initialPressureValues[i-1], Pfunction = [])  
for i in terminalList:    G.add_node(i, P = initialPressureValues[i-1], Pfunction = [], V = initialVolumeValues[i - firstLeaf], Vfunction = [])  

for i in nodeList:        G.add_edge(fatherList[i-1], i, Q = initialFlowValues[i-1], Qfunction = [], Diameter = 0)

def strahlerUpdate(lft, rgt):
    if(lft == rgt): 
        return (lft + 1)
    else:
        return max(lft,rgt)

for i in             terminalList : G[i][fatherList[i-1]]['Strahler'] = 1  # Setting the Strahler Number of the branch
for i in reversed(nonTerminalList): G[i][fatherList[i-1]]['Strahler'] = strahlerUpdate(G[i][leftSonList[i-1]]['Strahler'], G[i][rightSonList[i-1]]['Strahler'])
G[0][1]['Strahler'] = strahlerUpdate(G[1][2]['Strahler'], G[1][3]['Strahler'])
maxStrahler = G[0][1]['Strahler']

G[0][1]['Length'] = 0.05     #m Length of the longest branch
for i in nodeList[1:]: G[i][fatherList[i-1]]['Length'] = G[fatherList[i-1]][fatherList[fatherList[i-1]-1]]['Length'] * 0.4

strahlerToDiameter = xp.exp(xp.add(xp.multiply(xp.subtract(xp.arange(0, maxStrahler + 1), maxStrahler), math.log(RdS)), math.log(maxDiameter))) # diameter of branch with given Strahler number

for i in nodeList:
    G[fatherList[i-1]][i]['Diameter'] = strahlerToDiameter[G[fatherList[i-1]][i]['Strahler']]


def recomputeqConst(currentG): 
    aux, auy, lengths = zip(*currentG.edges.data('Length'))
    aux, auy, diameters = zip(*currentG.edges.data('Diameter'))
    return xp.divide(xp.multiply(xp.sqrt(lengths), 64 * correctionConstant * math.sqrt(miu * ro / (math.pi ** 3))), xp.power(diameters, 4))

computedVmin = 10   # L
computedVmax = 0.1  # L
tol = 0.05
timeList = []
PplList = []
volumeList = []
globalCycleCount = 0

N = (nodeCount - 1) + (firstLeaf - 1) + 2 * leafCount + 1 # number of equations
M = N                  # number of variables

J = spx.sparse.lil_matrix((N,M))

# equations: 0 to nodeCount-2: Pressure driven flow 
rows = xp.arange(0, nodeCount-1)
fatherNodePressures = xp.add(fatherList, 0)
nodePressures = xp.add(nodeList, 0)
J[rows, fatherNodePressures] = 1
J[rows, nodePressures] = -1

# equations: nodeCount-1 to nodeCount+firstLeaf-3: Conservation of flow
rows = xp.arange(nodeCount-1, nodeCount+firstLeaf-2)
nonTerminalNodeFlows = xp.add(nonTerminalList, nodeCount-1)
leftSonFlows = xp.add(leftSonList, nodeCount-1)
rightSonFlows = xp.add(rightSonList, nodeCount-1)
J[rows, nonTerminalNodeFlows] = 1
J[rows, leftSonFlows] = -1
J[rows, rightSonFlows] = -1

# equations: nodeCount+firstleaf-1 to 2*nodeCount-3: Expansion of acinar region by flow
rows = xp.arange(nodeCount+firstLeaf-2, 2*nodeCount-2)
terminalNodeVolumes = xp.add(terminalList, nodeCount+(nodeCount-1)-firstLeaf)
terminalNodeFlows = xp.add(terminalList, nodeCount-1)
J[rows, terminalNodeVolumes] = 1
J[rows, terminalNodeFlows] = -1000*0.1

# equations: 2*nodeCount-2 to 2*nodeCount-3+leafCount: Expansion of acinar region drive by flow equations 2
rows = xp.arange(2*nodeCount-2, 2*nodeCount-2+leafCount)
terminalNodePressures = xp.add(terminalList, 0)
J[rows, terminalNodePressures] = 1
J[rows, terminalNodeVolumes] = - (1/Compliance)

# equation: trachea pressure is atmospheric pressure
J[N-1, 0] = 1

#J = J.tocsr()
J = J.tocsc()

def computeCycle(currentG, cycleCount = 1, dt = 0.1, printValues = False, writeOver = False, getMaxVolumes = False, multiThreading = False): 
    # cycleCount = number of cycles
    # dt = time step size
    # printValues = keep track of volume/flows or not
    # writeOver = new volume overwrites old volume or not
    qConst = recomputeqConst(currentG)
    if(not multiThreading):
        global globalCycleCount, computedVmin, computedVmax
        globalCycleCount = globalCycleCount + cycleCount
        computedVmin = 10   # L
        computedVmax = 0.1  # L
        if(globalCycleCount % 100 == 0):
            print("DONE 100 CYCLES")

    iterations = int((cycleCount * breathLength)/dt) # Total number of iterations
    def PulmonarPressure(x): return Apl + Rpl * math.sin((2*math.pi*x)/breathLength) * fz # Pulmonar pressure at a given moment
    
    if(not multiThreading):
        if(printValues):
            global PplList, volumeList ,timeList
            volumeList = []
            timeList = []
            PplList = []
            for i in nodeList: currentG.nodes[i]['Pfunction'] = []
            for i in nodeList: currentG[i][fatherList[i-1]]['Qfunction'] = []
            for i in terminalList: currentG.nodes[i]['Vfunction'] = []
    aux, pressures =  zip(*currentG.nodes(data = 'P'))
    aux, auy, flows = zip(*currentG.edges.data('Q'))
    aux, volumes =  zip(*currentG.nodes(data = 'V'))
    oldx = xp.array(pressures + flows + volumes[firstLeaf:])

    maxVolumeList = []
    if(getMaxVolumes):
        maxVolumeList = volumes[firstLeaf:]

    nodeFlows = xp.add(nodeList, len(pressures)-1)
    
    def F(x):  return \
                    xp.concatenate((xp.subtract(xp.subtract(x[fatherNodePressures], x[nodePressures]), xp.multiply(qConst, xp.multiply(x[nodeFlows], xp.sqrt(xp.abs(x[nodeFlows]))))) \
                   ,xp.subtract(xp.subtract(x[nonTerminalNodeFlows], x[leftSonFlows]), x[rightSonFlows]) \
                   ,xp.subtract(xp.subtract(x[terminalNodeVolumes], xp.multiply(x[terminalNodeFlows], 1000*dt)), oldx[terminalNodeVolumes]) \
                   ,xp.subtract(xp.subtract(x[terminalNodePressures], xp.multiply(x[terminalNodeVolumes], 1/Compliance)), Ppl), xp.array([x[0]-Pat])))

    for step in range(0, iterations):
        newx = oldx
        Ppl = PulmonarPressure(step * dt)
        Fnewx = F(newx)
        while(max(xp.abs(Fnewx)) > eps):   
            nodeFlows = xp.add(nodeList, len(pressures)-1)
            J[xp.arange(0, nodeCount-1), nodeFlows] = xp.multiply(xp.multiply((-3/2), qConst), xp.sqrt(xp.abs(newx[nodeFlows])))
            #print("Count of non zeroes in Jacobian:", len(spx.sparse.find(J)[0]))
            #print("Size of  Jacobian:", N*M)
            #print("percentage of nonzeroes", 100 * len(spx.sparse.find(J)[0]) / (N*M))
            #dx = spx.sparse.linalg.spsolve(J, xp.negative(Fnewx))
            #dx, foundResult = spx.sparse.linalg.bicg(A = J, b = xp.negative(Fnewx), x0 = xp.zeros(newx.shape[0]),tol = 1e-10)
            #print(dx)
            B = spx.sparse.linalg.splu(J)
            dx = B.solve(xp.negative(Fnewx))
            newx = xp.add(newx, dx)
            #print(newx)
            Fnewx = F(newx)
            #print(Fnewx)
            #print("RESULT:", foundResult)
            #print(max(xp.abs(Fnewx)))
        #print("NEW")
        oldx = newx
        currentVolume = sum(oldx[terminalNodeVolumes])
        computedVmax = max(computedVmax, currentVolume)
        computedVmin = min(computedVmin, currentVolume)
        if(getMaxVolumes):
            maxVolumeList = xp.maximum(maxVolumeList, oldx[terminalNodeVolumes])
            
        if(not multiThreading):
            if(printValues):
                PplList.append(Ppl)
                timeList.append(step*dt)
                for i in nodeList: currentG.nodes[i]['Pfunction'].append(oldx[i])
                for i in nodeList: currentG[i][fatherList[i-1]]['Qfunction'].append(oldx[i+len(pressures)-1])
                for i in terminalList: currentG.nodes[i]['Vfunction'].append(oldx[i+len(pressures)+len(flows)-firstLeaf])
                volumeList.append(currentVolume)

    if(writeOver):
        nx.set_node_attributes(currentG, dict(zip(nodeList, oldx[nodeList])), name = 'P')
        nx.set_edge_attributes(currentG, dict(zip(zip(fatherList, nodeList), oldx[xp.add(nodeList, len(pressures)-1)])), name = 'Q')
        nx.set_node_attributes(currentG, dict(zip(terminalList, oldx[xp.add(terminalList, len(pressures)+len(flows)-firstLeaf)])), name = 'V')

    return maxVolumeList

def stabilizeVolume(currentG, multiThreading = False):
    aux, oldVolume =  zip(*currentG.nodes(data = 'V'))
    oldVolume = oldVolume[firstLeaf:]
    computeCycle(currentG = currentG, writeOver = True, multiThreading = multiThreading)
    aux, newVolume =  zip(*currentG.nodes(data = 'V'))
    newVolume = newVolume[firstLeaf:]
    while(max(xp.abs(xp.subtract(oldVolume, newVolume))) > tol):
        oldVolume = newVolume
        computeCycle(currentG = currentG, writeOver = True, multiThreading = multiThreading)
        aux, newVolume =  zip(*currentG.nodes(data = 'V'))
        newVolume = newVolume
       
print("APL is initially:", Apl)
print("RPL is initially:", Rpl)
print("Compliance is initially:", Compliance)
print("Initial volume is initially:", G.nodes[nodeCount/2]['V'])

def recomputeAplRpl(currentG):
    global Apl, Rpl
    computeCycle(currentG = currentG, writeOver = True)
    while(abs((Vdepth - (computedVmax - computedVmin)) / Vdepth) > tol or abs((FRC - computedVmin) / FRC) > tol):
        Apl = Apl * (FRC + Vdepth * (1/2)) / ((computedVmin + computedVmax) * (1/2))
        Rpl = Rpl * (Vdepth) / (computedVmax - computedVmin) 
        stabilizeVolume(currentG)

recomputeAplRpl(currentG = G)
aux, initialPressures =  zip(*G.nodes(data = 'P'))
initialPressures = initialPressures[1:]
aux, auy, initialFlows = zip(*G.edges.data('Q'))
aux, initialVolumes =    zip(*G.nodes(data = 'V'))
initialVolumes = initialVolumes[firstLeaf:]

if saveToFile:
    xp.save('initialInputs/initialInputs'+str(nodeCount)+'.npy', xp.concatenate(([Apl, Rpl], initialPressures, initialFlows, initialVolumes)))

print("APL is finally:", Apl)
print("RPL is finally:", Rpl)
print("Initial volume is finally:", G.nodes[nodeCount/2]['V'])
print("Global cycle count was:", globalCycleCount)

currentFigureNumber = 0 

def printCurrentLung(auxG, figureNumber = 0):
    print("Printing Lung: ", figureNumber)
    stabilizeVolume(currentG = auxG)
    computeCycle(currentG = auxG, cycleCount = 3, printValues = True)
    fig = plt.figure(figureNumber, figsize = (20, 10)) 
    fig.suptitle("Graphs for Healthy Lungs")
    fig.tight_layout()
    pressureAcinar = fig.add_subplot(2, 3, 1)
    pressureAcinar.title.set_text("Acinar pressure")
    pressureAcinar.set_ylabel("cmH2O")
    pressureAcinar.set_xlabel("s")
    pressureAcinar.plot(timeList, auxG.nodes[nodeCount/2]['Pfunction'])
    pressureAcinar.plot(timeList, xp.full(len(timeList), Pat))
    for i in range(1,6):
        pressureAcinar.axvline(x=i*2, color='black', linestyle='--')

    pressureAll = fig.add_subplot(2, 3, 2)
    pressureAll.title.set_text("Pressure at all depths")
    pressureAll.set_ylabel("cmH2O")
    pressureAll.set_xlabel("s")
    for i in oneEachDepth: pressureAll.plot(timeList, auxG.nodes[i]['Pfunction'])
    pressureAll.plot(timeList, xp.full(len(timeList), Pat))
    for i in range(1,6):
        pressureAll.axvline(x=i*2, color='black', linestyle='--')

    volumeAcinar = fig.add_subplot(2, 3, 3)
    volumeAcinar.title.set_text("Acinar volume")
    volumeAcinar.set_ylabel("L")
    volumeAcinar.set_xlabel("s")
    volumeAcinar.plot(timeList, auxG.nodes[nodeCount/2]['Vfunction'])
    for i in range(1,6):
        volumeAcinar.axvline(x=i*2, color='black', linestyle='--')

    branchFlow = fig.add_subplot(2, 3, 4)
    branchFlow.title.set_text("Branch flow")
    branchFlow.set_ylabel("m^3/s")
    branchFlow.set_xlabel("s")
    for i in oneEachDepth: plt.plot(timeList, auxG[math.floor(i/2)][i]['Qfunction'])
    for i in range(1,6):
        branchFlow.axvline(x=i*2, color='black', linestyle='--')

    pleuralPressure = fig.add_subplot(2, 3, 5)
    pleuralPressure.title.set_text("Pleural pressure")
    pleuralPressure.set_ylabel("cmH2O")
    pleuralPressure.set_xlabel("s")
    pleuralPressure.plot(timeList, PplList)
    for i in range(1,6):
        pleuralPressure.axvline(x=i*2, color='black', linestyle='--')

    global healthyVolumeList
    healthyVolumeList = volumeList
    print("Finished Printing Lung: ", figureNumber)
def computeUnhealthyLung(auxG, unhealthyBranch = 1, restrictionFactor = 2):
    print("Computing unhealthy lung "+str(unhealthyBranch))
    terminalHealthyBranch = int(nodeCount/2 + nodeCount/4)
    terminalUnhealthyBranch = int(nodeCount/2)

    auxG[unhealthyBranch][fatherList[unhealthyBranch-1]]['Diameter'] = auxG[unhealthyBranch][fatherList[unhealthyBranch-1]]['Diameter'] / restrictionFactor
    
    stabilizeVolume(auxG)
    computeCycle(currentG = auxG, cycleCount = 3, printValues = True, writeOver = False)

    global currentFigureNumber

    fig = plt.figure(currentFigureNumber, figsize = (20, 10))
    currentFigureNumber = currentFigureNumber + 1
    fig.suptitle("Graphs for Unhealthy Lung: "+str(unhealthyBranch))
    acinarPressure = fig.add_subplot(2, 3, 1)
    acinarPressure.title.set_text("Acinar pressure, Healthy vs unhealthy")
    acinarPressure.set_ylabel("cmH2O")
    acinarPressure.set_xlabel("s")
    acinarPressure.plot(timeList, auxG.nodes[terminalUnhealthyBranch]['Pfunction'], color = 'r')
    acinarPressure.plot(timeList, auxG.nodes[terminalHealthyBranch]['Pfunction'], color = 'g')
    for i in range(1,6):
        acinarPressure.axvline(x=i*2, color='black', linestyle='--')
        
    highPressure = fig.add_subplot(2, 3, 6)
    highPressure.title.set_text("Pressure, Left vs Right")
    highPressure.set_ylabel("cmH2O")
    highPressure.set_xlabel("s")
    highPressure.plot(timeList, auxG.nodes[2]['Pfunction'], color = 'r')
    highPressure.plot(timeList, auxG.nodes[3]['Pfunction'], color = 'g')
    for i in range(1,6):
        highPressure.axvline(x=i*2, color='black', linestyle='--')

    acinarVolume = fig.add_subplot(2, 3, 2)
    acinarVolume.title.set_text("Acinar volume, Healthy vs unhealthy")
    acinarVolume.set_ylabel("L")
    acinarVolume.set_xlabel("s")
    acinarVolume.plot(timeList, auxG.nodes[terminalUnhealthyBranch]['Vfunction'], color = 'r')
    acinarVolume.plot(timeList, auxG.nodes[terminalHealthyBranch]['Vfunction'], color = 'g')
    for i in range(1,6):
        acinarVolume.axvline(x=i*2, color='black', linestyle='--')

    totalVolume = fig.add_subplot(2, 3, 3)
    totalVolume.title.set_text("Total volume, Healthy lung vs unhealthy lung")
    totalVolume.set_ylabel("L")
    totalVolume.set_xlabel("s")
    totalVolume.plot(timeList, volumeList, color = 'r')
    totalVolume.plot(timeList, healthyVolumeList, color = 'g')
    for i in range(1,6):
        totalVolume.axvline(x=i*2, color='black', linestyle='--')

    flows = fig.add_subplot(2, 3, 4)
    flows.title.set_text("Left side vs right side Lung flow")
    flows.set_ylabel("m^3/s")
    flows.set_xlabel("s")
    flows.plot(timeList, auxG[1][2]['Qfunction'], color = 'r')
    flows.plot(timeList, auxG[1][3]['Qfunction'], color = 'g')
    for i in range(1,6):
        flows.axvline(x=i*2, color='black', linestyle='--')
    
    terminalFlows = fig.add_subplot(2, 3, 5)
    terminalFlows.title.set_text("Healthy vs Unhealthy Terminal flow")
    terminalFlows.set_ylabel("m^3/s")
    terminalFlows.set_xlabel("s")
    terminalFlows.plot(timeList, auxG[terminalUnhealthyBranch][fatherList[terminalUnhealthyBranch-1]]['Qfunction'], color = 'r')
    terminalFlows.plot(timeList, auxG[terminalHealthyBranch][fatherList[terminalHealthyBranch-1]]['Qfunction'], color = 'g')
    for i in range(1,6):
        terminalFlows.axvline(x=i*2, color='black', linestyle='--')
def computeGenericLung(auxG, restrictions = xp.full(nodeCount-1, 1), printLung = False, multiThreading = False):
    for i in nodeList:
        auxG[i][fatherList[i-1]]['Diameter'] = auxG[i][fatherList[i-1]]['Diameter'] * restrictions[i-1]
    stabilizeVolume(auxG, multiThreading = multiThreading)
    maxVolumeList = computeCycle(currentG = auxG, getMaxVolumes = True, multiThreading = multiThreading)
    if(printLung): 
        global currentFigureNumber
        printCurrentLung(auxG = auxG, figureNumber = currentFigureNumber)
        currentFigureNumber = currentFigureNumber + 1
    return maxVolumeList
    
import warnings
import time
import multiprocessing as mp
import logging
warnings.filterwarnings("ignore")

def computeMultipleGenericLung(auxG, restrictionsList, returnList):
    lungCount = len(restrictionsList)
    maxVolumesListList = []
    for i in range(lungCount):
        currentMaxVolumeList = computeGenericLung(auxG = auxG.copy(), restrictions = restrictionsList[i], printLung = False, multiThreading = True)
        maxVolumesListList.append(currentMaxVolumeList)
    #print(maxVolumesListList)
    returnList.extend(maxVolumesListList)


def multiThreadGenericLung(auxG, restrictions, threadCount = 4):
    start_time = time.time()
    rowCount =  len(restrictions)
    rowPerThread = int(rowCount/threadCount)
    threadList = []
    returnLists = []
    for i in range(0, threadCount):
        returnList = mp.Manager().list()
        returnLists.append(returnList)
        threadList.append(mp.Process(target=computeMultipleGenericLung, args=(auxG.copy(), restrictions[i*rowPerThread:(i+1)*rowPerThread], returnList)))
    for i in range(0, threadCount):
        threadList[i].start()
    for i in range(0, threadCount):
        threadList[i].join()
    end_time = time.time()
    #print(returnLists[0])
    print('Execution time = %.6f seconds' % (end_time-start_time))
    return np.concatenate(list(map(list, returnLists)), axis = 0)

#computeGenericLung(auxG = G.copy(), printLung = True)
#computeUnhealthyLung(auxG = G.copy(), unhealthyBranch = 2, restrictionFactor = 2)
#computeUnhealthyLung(auxG = G.copy(), unhealthyBranch = 4, restrictionFactor = 2)
#computeUnhealthyLung(auxG = G.copy(), unhealthyBranch = 8, restrictionFactor = 2)

