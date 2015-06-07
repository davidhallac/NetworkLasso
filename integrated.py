from snap import *
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
from multiprocessing import Pool


def solveX(data):
	inputs = int(data[data.size-1])
	lamb = data[data.size-2]
	rho = data[data.size-3]
	sizeData = data[data.size-4]
	x = data[0:inputs]
	a = data[inputs:(inputs + sizeData)]
	neighs = data[(inputs + sizeData):data.size-4]
	xnew = Variable(inputs,1)

	#REPLACE WITH ACTUAL OBJECTIVE FUNCTION HERE! Params: Xnew (unknown), a (side data at node)
	g = 0.5*square(norm(xnew - a))

	h = 0
	for i in range(neighs.size/(2*inputs+1)):
		weight = neighs[i*(2*inputs+1)]
		if(weight != 0):
			u = neighs[i*(2*inputs+1)+1:i*(2*inputs+1)+(inputs+1)]
			z = neighs[i*(2*inputs+1)+(inputs+1):(i+1)*(2*inputs+1)]
			h = h + rho/2*square(norm(xnew - z + u))
	objective = Minimize(50*g+50*h)
	constraints = []
	p = Problem(objective, constraints)
	result = p.solve()
	if(result == None):
		#CVXOPT scaling issue. Rarely happens (but occasionally does when running thousands of tests). Ignore for now
		objective = Minimize(51*g+52*h)
		p = Problem(objective, constraints)
		result = p.solve(verbose=False)
		if(result == None):
			print "SCALING BUG"
			objective = Minimize(52*g+50*h)
			p = Problem(objective, constraints)
			result = p.solve(verbose=False)
	return xnew.value, g.value

def solveZ(data):
	inputs = int(data[data.size-1])
	lamb = data[data.size-2]
	rho = data[data.size-3]
	x1 = data[0:inputs]
	x2 = data[inputs:2*inputs]
	u1 = data[2*inputs:3*inputs]
	u2 = data[3*inputs:4*inputs]
	weight = data[data.size-4]
	a = x1 + u1
	b = x2 + u2
	theta = max(1 - lamb*weight/(rho*LA.norm(a - b)+0.000001), 0.5) #So no divide by zero error
	z1 = theta*a + (1-theta)*b
	z2 = theta*b + (1-theta)*a
	znew = np.matrix(np.concatenate([z1, z2])).reshape(2*inputs,1)
	return znew

def solveU(data):
	leng = data.size-1
	u = data[0:leng/3]
	x = data[leng/3:2*leng/3]
	z = data[(2*leng/3):leng]
	rho = data[data.size-1]
	return u + (x - z)

def runADMM(G1, sizeOptVar, sizeData, lamb, rho, numiters, x, u, z, a, edgeWeights):

	nodes = G1.GetNodes()
	edges = G1.GetEdges()

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

		temp = np.concatenate((x,a,neighs,np.tile([sizeData,rho,lamb,sizeOptVar], (nodes,1)).transpose()), axis=0)
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
		temp = np.concatenate((xtemp,utemp,ztemp,np.reshape(weightsList, (-1,edges)),np.tile([rho,lamb,sizeOptVar], (edges,1)).transpose()), axis=0)
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


		#Stopping criterion - p19 of ADMM paper
		epri = sqp*eabs + erel*max(LA.norm(np.dot(A,x.transpose()), 'fro'), LA.norm(z, 'fro'))
		edual = sqn*eabs + erel*LA.norm(np.dot(A.transpose(),u.transpose()), 'fro')
		r = LA.norm(np.dot(A,x.transpose()) - z.transpose(),'fro')
		s = s

		print r, epri, s, edual
		iters = iters + 1

	pool.close()
	pool.join()


	return (x,u,z,0,0)

def main():

	#Set parameters
	rho = 0.1 #ADMM parameter
	numiters = 25 #Max number of ADMM iterations at each step
	thresh = 3 #maximum lambda value
	lambInit = 0.5 #Initial non-zero value of lambda to start at
	updateVal = 1.25 #Amount to update lambda each iteration

	#Size of x, the variable we solve for
	sizeOptVar = 5
	#Amount of side information at each node
	sizeData = 5


	#Generate graph, edge weights.
	#REPLACE WITH ACTUAL GRAPH HERE
	n = 10
	e = 25
	np.random.seed(2)
	G1 = GenRndGnm(PUNGraph, n, e)
	edgeWeights = TIntPrFltH()
	for EI in G1.Edges():
		temp = TIntPr(EI.GetSrcNId(), EI.GetDstNId())
		edgeWeights.AddDat(temp, 1)

	#Get relevant graph info
	nodes = G1.GetNodes()
	edges = G1.GetEdges()

	#Generate side information
	#REPLACE WITH ACTUAL SIDE INFORMATION HERE
	a = np.random.randn(sizeData, nodes)

	#Initialize variables to 0. Get relevant info
	x = np.zeros((sizeOptVar,nodes))
	u = np.zeros((sizeOptVar,2*edges))
	z = np.zeros((sizeOptVar,2*edges))

	#Run regularization path
	lamb = 0
	while(lamb <= thresh or lamb == 0):
		(x, u, z, pl1, pl2) = runADMM(G1, sizeOptVar, sizeData, lamb, rho + lamb/10, numiters, x, u ,z, a, edgeWeights)
		print "Lambda = ", lamb
		if(lamb == 0):
			lamb = lambInit
		else:
			lamb = lamb * updateVal







if __name__ == '__main__':
	main()
