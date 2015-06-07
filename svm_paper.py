from snap import *
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
from multiprocessing import Pool
#Plotting
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
	numtests = int(data[data.size-5])
	c = data[data.size-6]
	x = data[0:inputs]
	rawData = data[inputs:(inputs + sizeData)]
	neighs = data[(inputs + sizeData):data.size-6]
	xnew = Variable(inputs,1)

	x_train = rawData[0:numtests*inputs]
	y_train = rawData[numtests*inputs: numtests*(inputs+1)]
	a = Variable(inputs,1)
	epsil = Variable(numtests,1)
	constraints = [epsil >= 0]
	g = c*norm(epsil,1)
	for i in range(inputs - 1):
		g = g + 0.5*square(a[i])
	for i in range(numtests):
		temp = np.asmatrix(x_train[i*inputs:(i+1)*inputs])
		constraints = constraints + [y_train[i]*(temp*a) >= 1 - epsil[i]]
	f = 0
	for i in range(neighs.size/(2*inputs+1)):
		weight = neighs[i*(2*inputs+1)]
		if(weight != 0):
			u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
			z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
			f = f + rho/2*square(norm(a - z + u))
	objective = Minimize(50*g + 50*f)
	p = Problem(objective, constraints)
	result = p.solve()
	if(result == None):
		#CVXOPT scaling issue. Rarely happens (but occasionally does when running thousands of tests)
		objective = Minimize(50*g+51*f)
		p = Problem(objective, constraints)
		result = p.solve(verbose=False)
		if(result == None):
			print "SCALING BUG"
			objective = Minimize(52*g+50*f)
			p = Problem(objective, constraints)
			result = p.solve(verbose=False)
	return a.value, g.value














def runADMM(G1, sizeOptVar, sizeData, lamb, rho, numiters, x, u, z, a, edgeWeights, numtests, useConvex, c, epsilon):

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
		temp = np.concatenate((x,a,neighs,np.tile([c, numtests,sizeData,rho,lamb,sizeOptVar], (nodes,1)).transpose()), axis=0)
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
			# else:
			# 	print "FAILED AT ITERATION ", iters, "; Obj = ", tempObj, "; Initial = ", initTemp

			if(iters == numiters - 1 and numiters < maxNonConvexIters):
				if(bestObj == initObj):
					numiters = numiters+1

		#Stopping criterion - p19 of ADMM paper
		epri = sqp*eabs + erel*max(LA.norm(np.dot(A,x.transpose()), 'fro'), LA.norm(z, 'fro'))
		edual = sqn*eabs + erel*LA.norm(np.dot(A.transpose(),u.transpose()), 'fro')
		r = LA.norm(np.dot(A,x.transpose()) - z.transpose(),'fro')
		s = s

		#print r, epri, s, edual
		iters = iters + 1

	pool.close()
	pool.join()

	if(useConvex == 1):
		return (x,u,z,0,0)
	else:
		return (bestx,bestu,bestz,0,0)















def main():

	#Set parameters
	useConvex = 1 #1 = true, 0 = false
	rho = 0.0001 
	numiters = 50
	thresh = -1#10
	lamb = 0.0
	startVal = 0.001
	useMult = 1 #1 for mult, 0 for add
	addUpdateVal = 0.1 
	multUpdateVal = 1.2


	#Graph Information
	nodes = 1000
	#Number of partitions
	partitions = 20
	samepart = 0.5
	diffpart = 0.01
	#Size of x
	sizeOptVar = 51 #Includes 1 for constant offset!
	#C in SVM
	c = 0.75
	#Non-convex variable
	epsilon = 0.01
	#Training set size
	numtests = 25
	testSetSize = 10


	#Generate graph, edge weights
	np.random.seed(2)
	G1 = TUNGraph.New()
	for i in range(nodes):
		G1.AddNode(i)
	sizepart = nodes/partitions
	correctedges = 0
	for NI in G1.Nodes():
		for NI2 in G1.Nodes():
			if(NI.GetId() < NI2.GetId()):				
				if ((NI.GetId()/sizepart) == (NI2.GetId()/sizepart)):
					#Same partition, edge w.p 0.5
					if(np.random.random() >= 1-samepart):
						G1.AddEdge(NI.GetId(), NI2.GetId())
						correctedges = correctedges+1
				else:
					if(np.random.random() >= 1-diffpart):
						G1.AddEdge(NI.GetId(), NI2.GetId())

	edges = G1.GetEdges()

	edgeWeights = TIntPrFltH()
	for EI in G1.Edges():
		temp = TIntPr(EI.GetSrcNId(), EI.GetDstNId())
		edgeWeights.AddDat(temp, 1)

	#Generate side information
	a_true = np.random.randn(sizeOptVar, partitions)
	v = np.random.randn(numtests,nodes)
	vtest = np.random.randn(testSetSize,nodes)

	trainingSet = np.random.randn(numtests*(sizeOptVar+1), nodes) #First all the x_train, then all the y_train below it
	for i in range(numtests):
		trainingSet[(i+1)*sizeOptVar - 1, :] = 1 #Constant offset
	for i in range(nodes):
		a_part = a_true[:,i/sizepart]
		for j in range(numtests):
			trainingSet[numtests*sizeOptVar+j,i] = np.sign([np.dot(a_part.transpose(), trainingSet[j*sizeOptVar:(j+1)*sizeOptVar,i])+v[j,i]])

	(x_test,y_test) = (np.random.randn(testSetSize*sizeOptVar, nodes), np.zeros((testSetSize, nodes)))
	for i in range(testSetSize):
		x_test[(i+1)*sizeOptVar - 1, :] = 1 #Constant offset
	for i in range(nodes):
		a_part = a_true[:,i/sizepart]
		for j in range(testSetSize):
			y_test[j,i] = np.sign([np.dot(a_part.transpose(), x_test[j*sizeOptVar:(j+1)*sizeOptVar,i])+vtest[j,i]])

	sizeData = trainingSet.shape[0]

	nodes = G1.GetNodes()
	edges = G1.GetEdges()
	print nodes, edges, correctedges/float(edges), 1 - float(correctedges)/edges
	print GetBfsFullDiam(G1, 1000, False);

	#Initialize variables to 0
	x = np.zeros((sizeOptVar,nodes))
	u = np.zeros((sizeOptVar,2*edges))
	z = np.zeros((sizeOptVar,2*edges))

	#Run regularization path
	[plot1, plot2, plot3] = [TFltV(), TFltV(), TFltV()]	
	while(lamb <= thresh or lamb == 0):
		(x, u, z, pl1, pl2) = runADMM(G1, sizeOptVar, sizeData, lamb, rho + math.sqrt(lamb), numiters, x, u ,z, trainingSet, edgeWeights, numtests, useConvex, c, epsilon)
		print "Lambda = ", lamb

		#Get accuracy
		(right, total) = (0, testSetSize*nodes)
		a_pred = x
		for i in range(nodes):
			temp = a_pred[:,i]
			for j in range(testSetSize):
				pred = np.sign([np.dot(temp.transpose(), x_test[j*sizeOptVar:(j+1)*sizeOptVar,i])])
				if(pred == y_test[j,i]):
					right = right + 1
		accuracy = right / float(total)
		cons = 0
		for i in range(edges):
			if(np.all(z[:,2*i] == z[:,2*i + 1])):
				cons = cons + 1
		consensus = cons / float(edges)
		print accuracy, consensus

		plot1.Add(lamb)
		plot2.Add(accuracy)
		plot3.Add(consensus)
		if(lamb == 0):
			lamb = startVal
		elif(useMult == 1):
			lamb = lamb*multUpdateVal
		else:
			lamb = lamb + addUpdateVal


	#Print/Save plot of results
	if(thresh > 0):
		pl1 = np.array(plot1)
		pl2 = np.array(plot2)
		plt.plot(pl1, pl2)
		plt.xscale('log')
		plt.xlabel(r'$\lambda$')
		plt.ylabel('Prediction Accuracy')
		if(useConvex == 1):
			plt.savefig('image_svm_convex',bbox_inches='tight')
		else:
			plt.savefig('image_svm_nonconvex',bbox_inches='tight')


		#Plot of clustering
		plt.figure()
		pl3 = np.array(plot3)
		plt.plot(pl1, pl3)
		plt.xscale('log')
		plt.xlabel(r'$\lambda$')
		plt.ylabel('\% of edges in Consensus')
		if(useConvex == 1):
			plt.savefig('consensus_svm_convex',bbox_inches='tight')
			np.savetxt('text_svm_convex.out', (pl1, pl2, pl3), delimiter=',', fmt='%1.4f')
		else:
			plt.savefig('consensus_svm_nonconvex',bbox_inches='tight')
			np.savetxt('text_svm_nonconvex.out', (pl1, pl2, pl3), delimiter=',', fmt='%1.4f')

	#Draw Graph
	#DrawGViz(G1, gvlNeato, "graph_undirected.png", "graph 2", True)



if __name__ == '__main__':
	main()
