from snap import *
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
from multiprocessing import Pool
#Plotting
import csv
import os    
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text',usetex=True)
import matplotlib.pyplot as plt
#Other function in this folder
from z_u_solvers import solveZ, solveU

def solveX(data):
	inputs = int(data[data.size-1])
	lamb = data[data.size-2]
	rho = data[data.size-3]
	sizeData = data[data.size-4]
	tolerance = 3
	x = data[0:inputs]
	a = data[inputs:(inputs + sizeData)]
	neighs = data[(inputs + sizeData):data.size-4]
	xnew = Variable(inputs,1)

	mu = data[data.size-5]

	#Fill in objective function here! Params: Xnew (unknown), a (side data at node)


	g = square(norm(xnew - a)) + mu*(norm(xnew))

	h = 0
	for i in range(neighs.size/(2*inputs+1)):
		weight = neighs[i*(2*inputs+1)]
		if(weight != 0):
			u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
			z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
			h = h + rho/2*square(norm(xnew - z + u))
	objective = Minimize(5*g+5*h)
	constraints = []
	p = Problem(objective, constraints)
	result = p.solve()
	if(result == None):
		#Todo: CVXOPT scaling issue
		objective = Minimize(g+1.001*h)
		p = Problem(objective, constraints)
		result = p.solve(verbose=False)
		print "SCALING BUG"
	return xnew.value, g.value

def runADMM(G1, sizeOptVar, sizeData, lamb, rho, numiters, x, u, z, a, edgeWeights, useConvex, epsilon, mu):

	nodes = G1.GetNodes()
	edges = G1.GetEdges()

	maxNonConvexIters = 6*numiters

	#Find max degree of graph; hash the nodes
	(maxdeg, counter) = (0, 0)
	node2mat = TIntIntH()
	for NI in G1.Nodes():
		maxdeg = max(maxdeg, NI.GetDeg())
		node2mat.AddDat(NI.GetId(), counter)
		counter = counter + 1

	#Stopping criteria
	eabs = math.pow(10,-2)
	erel = math.pow(10,-3)
	(r, s, epri, edual, counter) = (1,1,0,0,0)
	A = np.zeros((2*edges, nodes))
	for EI in G1.Edges():
		A[2*counter,node2mat.GetDat(EI.GetSrcNId())] = 1
		A[2*counter+1, node2mat.GetDat(EI.GetDstNId())] = 1
		counter = counter+1
	(sqn, sqp) = (math.sqrt(nodes*sizeOptVar), math.sqrt(2*sizeOptVar*edges))

	#Non-convex case - keeping track of best point so far
	bestx = x
	bestu = u
	bestz = z
	bestObj = 0
	cvxObj = 10000000*np.ones((1, nodes))
	if(useConvex != 1):
		#Calculate objective
		for i in range(G1.GetNodes()):
			bestObj = bestObj + cvxObj[0,i]
		for EI in G1.Edges():
			weight = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
			edgeDiff = LA.norm(x[:,node2mat.GetDat(EI.GetSrcNId())] - x[:,node2mat.GetDat(EI.GetDstNId())])
			bestObj = bestObj + lamb*weight*math.log(1 + edgeDiff / epsilon)
		initObj = bestObj

	#Run ADMM
	iters = 0
	maxProcesses =  80
	pool = Pool(processes = min(max(nodes, edges), maxProcesses))
	while(iters < numiters and (r > epri or s > edual or iters < 1)):

		#x-update
		neighs = np.zeros(((2*sizeOptVar+1)*maxdeg,nodes))
		edgenum = 0
		numSoFar = TIntIntH()
		for EI in G1.Edges():
			if (not numSoFar.IsKey(EI.GetSrcNId())):
				numSoFar.AddDat(EI.GetSrcNId(), 0)
			counter = node2mat.GetDat(EI.GetSrcNId())
			counter2 = numSoFar.GetDat(EI.GetSrcNId())
 			neighs[counter2*(2*sizeOptVar+1),counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
 			neighs[counter2*(2*sizeOptVar+1)+1:counter2*(2*sizeOptVar+1)+(sizeOptVar+1),counter] = u[:,2*edgenum] 
 			neighs[counter2*(2*sizeOptVar+1)+(sizeOptVar+1):(counter2+1)*(2*sizeOptVar+1),counter] = z[:,2*edgenum]
			numSoFar.AddDat(EI.GetSrcNId(), counter2+1)

			if (not numSoFar.IsKey(EI.GetDstNId())):
				numSoFar.AddDat(EI.GetDstNId(), 0)
			counter = node2mat.GetDat(EI.GetDstNId())
			counter2 = numSoFar.GetDat(EI.GetDstNId())
 			neighs[counter2*(2*sizeOptVar+1),counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
 			neighs[counter2*(2*sizeOptVar+1)+1:counter2*(2*sizeOptVar+1)+(sizeOptVar+1),counter] = u[:,2*edgenum+1] 
 			neighs[counter2*(2*sizeOptVar+1)+(sizeOptVar+1):(counter2+1)*(2*sizeOptVar+1),counter] = z[:,2*edgenum+1]
			numSoFar.AddDat(EI.GetDstNId(), counter2+1)
			edgenum = edgenum+1
		temp = np.concatenate((x,a,neighs,np.tile([mu, sizeData,rho,lamb,sizeOptVar], (nodes,1)).transpose()), axis=0)
		values = pool.map(solveX, temp.transpose())
		newx = np.array(values)[:,0].tolist()
		newcvxObj = np.array(values)[:,1].tolist()
		x = np.array(newx).transpose()[0]
		cvxObj = np.reshape(np.array(newcvxObj), (-1, nodes))

		#z-update
		ztemp = z.reshape(2*sizeOptVar, edges, order='F')
		utemp = u.reshape(2*sizeOptVar, edges, order='F')
		xtemp = np.zeros((sizeOptVar, 2*edges))
		counter = 0
		weightsList = np.zeros((1, edges))
		for EI in G1.Edges():
			xtemp[:,2*counter] = np.array(x[:,node2mat.GetDat(EI.GetSrcNId())])
			xtemp[:,2*counter+1] = x[:,node2mat.GetDat(EI.GetDstNId())]
			weightsList[0,counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
			counter = counter+1
		xtemp = xtemp.reshape(2*sizeOptVar, edges, order='F')
		temp = np.concatenate((xtemp,utemp,ztemp,np.reshape(weightsList, (-1,edges)),np.tile([epsilon, useConvex, rho,lamb,sizeOptVar], (edges,1)).transpose()), axis=0)
		newz = pool.map(solveZ, temp.transpose())
		ztemp = np.array(newz).transpose()[0]
		ztemp = ztemp.reshape(sizeOptVar, 2*edges, order='F')
		s = LA.norm(rho*np.dot(A.transpose(),(ztemp - z).transpose())) #For dual residual
		z = ztemp

		#u-update
		(xtemp, counter) = (np.zeros((sizeOptVar, 2*edges)), 0)
		for EI in G1.Edges():
			xtemp[:,2*counter] = np.array(x[:,node2mat.GetDat(EI.GetSrcNId())])
			xtemp[:,2*counter+1] = x[:,node2mat.GetDat(EI.GetDstNId())]
			counter = counter + 1
		temp = np.concatenate((u, xtemp, z, np.tile(rho, (1,2*edges))), axis=0)
		newu = pool.map(solveU, temp.transpose())
		u = np.array(newu).transpose()

		#Update best objective (for non-convex)
		if(useConvex != 1):
			tempObj = 0
			#Calculate objective
			for i in range(G1.GetNodes()):
				tempObj = tempObj + cvxObj[0,i]
			initTemp = tempObj
			for EI in G1.Edges():
				weight = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
				edgeDiff = LA.norm(x[:,node2mat.GetDat(EI.GetSrcNId())] - x[:,node2mat.GetDat(EI.GetDstNId())])
				tempObj = tempObj + lamb*weight*math.log(1 + edgeDiff / epsilon)

			#Update best variables
			if(tempObj < bestObj or bestObj == -1):
				bestx = x
				bestu = u
				bestz = z
				bestObj = tempObj
				print "Iteration ", iters, "; Obj = ", tempObj, "; Initial = ", initTemp

			if(iters == numiters - 1 and numiters < maxNonConvexIters):
				if(bestObj == initObj):
					numiters = numiters+1

		#Stopping criterion - p19 of ADMM paper
		epri = sqp*eabs + erel*max(LA.norm(np.dot(A,x.transpose()), 'fro'), LA.norm(z, 'fro'))
		edual = sqn*eabs + erel*LA.norm(np.dot(A.transpose(),u.transpose()), 'fro')
		r = LA.norm(np.dot(A,x.transpose()) - z.transpose(),'fro')
		s = s

		print r, epri, s, edual
		iters = iters + 1

	pool.close()
	pool.join()

	if(useConvex == 1):
		return (x,u,z,0,0)
	else:
		return (bestx,bestu,bestz,0,0)




















def main():

	#ANOMALY-SPECIFIC VARIABLES
	eventThresh = 0.01
	mu = 2.2
	numTrials = 105
	multUpdate = 1.04
	results = np.zeros((numTrials,6))
	threshSched = np.zeros((numTrials,1))







	#Set parameters
	useConvex = 0
	rho = 0.001
	numiters = 50
	thresh = 6
	lamb = 6
	startVal = 0.01
	useMult = 1 #1 for mult, 0 for add
	addUpdateVal = 0.1 
	multUpdateVal = 1.1

	#Size of x
	sizeOptVar = 2
	#Size of side information at each node
	sizeData = 2
	#Non-convex vars
	epsilon = 0.01


	#Generate graph, edge weights
	file = open("Data/CalIt2.csv", "rU")
	dataset = TIntFltVH()
	G1 = TUNGraph.New()
	counter = 0
	#for line in file:
	while True:
		line = file.readline() #7 --outflow
		if not line: 
			break
		outward = float(line.split(",")[3])
		line = file.readline() #9 --inflow
		inward = float(line.split(",")[3])

		tempData = TFltV()
		tempData.Add(outward)
		tempData.Add(inward)
		dataset.AddDat(counter, tempData)

		G1.AddNode(counter)
		counter = counter + 1

	#Build linear graph
	for NI in G1.Nodes():
		if (NI.GetId() > 0):
			G1.AddEdge(NI.GetId(), NI.GetId()-1)
		
	#Edge weights
	edgeWeights = TIntPrFltH()
	for EI in G1.Edges():
		temp = TIntPr(EI.GetSrcNId(), EI.GetDstNId())
		edgeWeights.AddDat(temp, 1)

	nodes = G1.GetNodes()
	edges = G1.GetEdges()

	#Save side information
	a = np.zeros((2, nodes))
	for NI in G1.Nodes():
		a[0,NI.GetId()] = dataset.GetDat(NI.GetId())[0]
		a[1,NI.GetId()] = dataset.GetDat(NI.GetId())[1]
		#print dataset.GetDat(NI.GetId())[0], dataset.GetDat(NI.GetId())[1] 
	
	#Get baseline for each day/time (median)
	baseline = np.zeros((2,48*7))
	for i in range(48*7):
		(counter, counter2) = (0,0)
		(temp, temp2) = (np.zeros((15,1)),np.zeros((15,1)))
		for j in range(15):
			counter = counter + a[0, i + 48*7*j]
			counter2 = counter2 + a[1, i + 48*7*j]
			temp[j] = a[0,i + 48*7*j]
			temp2[j] = a[1,i + 48*7*j]
		baseline[0,i] = np.median(temp)#counter / 15
		baseline[1,i] = np.median(temp2)#counter2 / 15


	#Subtract Baseline from a
	for i in range(nodes):
		a[0,i] = a[0,i] - baseline[0, i  % (48*7)]
		a[1,i] = a[1,i] - baseline[1, i  % (48*7)]

	#Initialize variables to 0
	x = np.zeros((sizeOptVar,nodes))
	u = np.zeros((sizeOptVar,2*G1.GetEdges()))
	z = np.zeros((sizeOptVar,2*G1.GetEdges()))

	# #Run regularization path
	# while(lamb <= thresh or lamb == 0):
	# #while(False):
	# 	(x, u, z, pl1, pl2) = runADMM(G1, sizeOptVar, sizeData, lamb, rho + math.sqrt(lamb)/2, numiters, x, u ,z, a, edgeWeights, useConvex, epsilon, mu)
	# 	print "Lambda = ", lamb


	# 	if(lamb == 0):
	# 		lamb = startVal
	# 	elif(useMult == 1):
	# 		lamb = lamb*multUpdateVal
	# 	else:
	# 		lamb = lamb + addUpdateVal


	for q in range(numTrials):

		(x, u, z, pl1, pl2) = runADMM(G1, sizeOptVar, sizeData, lamb, rho + math.sqrt(lamb)/2, numiters, x, u ,z, a, edgeWeights, useConvex, epsilon, mu)

		#Compare to events
		file = open("Data/CalIt2Events.csv", "rU")
		events = TIntPrV()	
		for line in file:
			if(not line.split(",")[0]):
				break
			events.Add(TIntPr(float(line.split(",")[4]), float(line.split(",")[5])))	

		truth = np.zeros((2,nodes))
		for meeting in events:
			start = meeting.GetVal1()
			end = meeting.GetVal2()
			counter = start
			while (counter <= end):
				truth[0,counter] = truth[0,counter] + 10
				counter = counter + 1


		#print results
		plt.plot(range(nodes), x[0,:])
		plt.plot(range(nodes), x[1,:], color='r')
		plt.plot(range(nodes), truth[0,:], color='g',linestyle='--')
		plt.xlim([0,5040])
		plt.savefig('image_svm_convex',bbox_inches='tight')	

		#Predict events
		counter = 0
		maxLength = 0
		for i in range(nodes-1):
			if (x[0,i] + x[1,i] >= eventThresh and x[0,i-1] + x[1,i-1] < eventThresh):
			 	beginning = i
			if (x[0,i] + x[1,i] >= eventThresh and x[0,i+1] + x[1,i+1] < eventThresh):	
				end  = i
				#print "Event ", counter, " starts at ", beginning, "and is length ", i - beginning + 1
				maxLength = max(maxLength, i-beginning+1)
				counter = counter + 1
		#print maxLength, " = maximum length"
		#print counter, " events predicted"

		numPred = counter

		counter = 0
		start = 0
		correct = 0
		for i in range(nodes):		

			if (x[0,i] + x[1,i] >= eventThresh):
				counter = counter + 1
				#Check if it was correctly counted
				if(sum(truth[0,i-1:i+1]) > 0):
					correct = correct + 1

		# print counter, " timestamps triggered"
		# print correct, " correct answers"

		timestampsPred = counter
		timestampsCorr = correct


		#See how many of the 30 events were detected
		numevents = 0
		for meeting in events:
			start = meeting.GetVal1()
			end = meeting.GetVal2()
			counter = start
			while (counter <= end):
				if(x[0,counter] + x[1,counter] >= eventThresh):
					numevents = numevents + 1
					break
				counter = counter + 1
				#if (counter > end):
					#print "Missed event from ", start, " to ", end

		#print numevents, " events detected"


		results[q,:] = [mu, numevents, numPred, timestampsCorr, timestampsPred, maxLength]
		threshSched[q,0] = mu

		#Update mu
		mu = multUpdate*mu

	np.set_printoptions(formatter={'float': '{: 0.7f}'.format})
	print(threshSched)
	np.set_printoptions(suppress=True)
	print results

	np.savetxt("results.csv", results, delimiter=",")

if __name__ == '__main__':
	main()
