from snap import *
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
from multiprocessing import Pool

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def solveX(data):
	inputs = int(data[data.size-1])
	lamb = data[data.size-2]
	rho = data[data.size-3]
	sizeData = data[data.size-4]
	x = data[0:inputs]
	a = data[inputs:(inputs + sizeData)]
	neighs = data[(inputs + sizeData):data.size-4]
	xnew = Variable(inputs,1)
	#Fill in objective function here! Params: Xnew (unknown), a (side data at node)
	g = 0.5*square(norm(xnew[0] - a[4]/100000))
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
	return xnew.value

def solveZ(data):
	inputs = int(data[data.size-1])
	lamb = data[data.size-2]
	rho = data[data.size-3]
	useConvex = data[data.size-4]
	epsilon = data[data.size-5]
	weight = data[data.size-6]
	x1 = data[0:inputs]
	x2 = data[inputs:2*inputs]
	u1 = data[2*inputs:3*inputs]
	u2 = data[3*inputs:4*inputs]
	a = x1 + u1
	b = x2 + u2
	(z1, z2) = (0,0)
	if(useConvex == 1):
		theta = max(1 - lamb*weight/(rho*LA.norm(a - b)+0.000001), 0.5) #So no divide by zero error
		z1 = theta*a + (1-theta)*b
		z2 = theta*b + (1-theta)*a
	else: #Non-convex version
		d = LA.norm(a-b)
		c = lamb*weight

		if(math.pow(rho,2)*math.pow(d+epsilon,2) - 8*rho*c >= 0):
			theta1 = (-rho*(d+epsilon) + math.sqrt(math.pow(rho,2)*math.pow(d+epsilon,2) - 8*rho*c)) / (4*rho*d+0.000001)
			theta1 = min(max(theta1,0),0.5)
			phi = math.log(1 + d*(1-2*theta1)/epsilon)
			objective1 = c*phi + rho*math.pow(d,2)*(math.pow(theta1,2))

			theta2 = (-rho*(d+epsilon) - math.sqrt(math.pow(rho,2)*math.pow(d+epsilon,2) - 8*rho*c)) / (4*rho*d+0.000001)
			theta2 = min(max(theta2,0),0.5)
			phi = math.log(1 + d*(1-2*theta2)/epsilon)
			objective2 = c*phi + rho*math.pow(d,2)*(math.pow(theta2,2))

			objective3 = rho/4*math.pow(LA.norm(a-b),2)

			if(min(objective1, objective2, objective3) == objective1):
				theta = min(max(theta1,0),0.5)
			elif(min(objective1, objective2, objective3) == objective2):
				theta = min(max(theta2,0),0.5)
			else:
				theta = 0.5
			z1 = (1-theta)*a + theta*b
			z2 = theta*a + (1-theta)*b
		else: #No real roots, use theta = 0.5
			(z1, z2) = (0.5*a + 0.5*b, 0.5*a + 0.5*b)
			
	znew = np.matrix(np.concatenate([z1, z2])).reshape(2*inputs,1)
	return znew

def solveU(data):
	leng = data.size-1
	u = data[0:leng/3]
	x = data[leng/3:2*leng/3]
	z = data[(2*leng/3):leng]
	rho = data[data.size-1]
	return u + (x - z)

def runADMM(G1, sizeOptVar, sizeData, lamb, rho, numiters, x, u, z, a, edgeWeights, useConvex, epsilon):

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

	#Non-convex case - keeping track of best point so far
	bestx = x
	bestu = u
	bestz = z
	bestObj = 0
	if(useConvex != 1):
		#Calculate objective
		for i in range(G1.GetNodes()):
			bestObj = bestObj + 0.5*math.pow(LA.norm(x[0,i] - a[4,i]/100000),2)
		for EI in G1.Edges():
			weight = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
			edgeDiff = LA.norm(x[0,node2mat.GetDat(EI.GetSrcNId())] - x[0,node2mat.GetDat(EI.GetDstNId())])
			bestObj = bestObj + lamb*weight*math.log(1 + edgeDiff / epsilon)

	#Run ADMM
	iters = 0
	maxProcesses =  80
	pool = Pool(processes = min(max(nodes, edges), maxProcesses))
	while(iters < numiters and (r > epri or s > edual or iters < 1)):

		#x-update
		(neighs, counter) = (np.zeros(((2*sizeOptVar+1)*maxdeg,nodes)), 0)
		for NI in G1.Nodes():
			counter2 = 0
			edgenum = 0
			for EI in G1.Edges():
				if (EI.GetSrcNId() == NI.GetId()):
					neighs[counter2*(2*sizeOptVar+1),counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
					neighs[counter2*(2*sizeOptVar+1)+1:counter2*(2*sizeOptVar+1)+(sizeOptVar+1),counter] = u[:,2*edgenum] #u_ij 
					neighs[counter2*(2*sizeOptVar+1)+(sizeOptVar+1):(counter2+1)*(2*sizeOptVar+1),counter] = z[:,2*edgenum] #z_ij
					counter2 = counter2 + 1
				elif (EI.GetDstNId() == NI.GetId()):
					neighs[counter2*(2*sizeOptVar+1),counter] = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
					neighs[counter2*(2*sizeOptVar+1)+1:counter2*(2*sizeOptVar+1)+(sizeOptVar+1),counter] = u[:,2*edgenum+1] #u_ij 
					neighs[counter2*(2*sizeOptVar+1)+(sizeOptVar+1):(counter2+1)*(2*sizeOptVar+1),counter] = z[:,2*edgenum+1] #z_ij
					counter2 = counter2 + 1
				edgenum = edgenum+1
			counter = counter + 1
		temp = np.concatenate((x,a,neighs,np.tile([sizeData,rho,lamb,sizeOptVar], (nodes,1)).transpose()), axis=0)
		newx = pool.map(solveX, temp.transpose())
		x = np.array(newx).transpose()[0]

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
				tempObj = tempObj + 0.5*math.pow(LA.norm(x[0,i] - a[4,i]/100000),2)
			for EI in G1.Edges():
				weight = edgeWeights.GetDat(TIntPr(EI.GetSrcNId(), EI.GetDstNId()))
				edgeDiff = LA.norm(x[0,node2mat.GetDat(EI.GetSrcNId())] - x[0,node2mat.GetDat(EI.GetDstNId())])
				tempObj = tempObj + lamb*weight*math.log(1 + edgeDiff / epsilon)
			#Update best variables
			if(tempObj < bestObj):
				bestx = x
				bestu = u
				bestz = z
				bestObj = tempObj
				print "Iteration ", iters, "; Obj = ", tempObj

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
	useConvex = 0
	rho = 0.001
	numiters = 50
	thresh = 10
	lamb = 0.0
	updateVal = 0.1
	numNeighs = 5
	#Test/Validation Set Information
	testSetSize = 100
	validationSetSize = 100
	numNewNeighs = 5 #For test/validation nodes
	#Size of x
	sizeOptVar = 2
	#Size of side information at each node
	sizeData = 5
	#Non-convex vars
	epsilon = 0.1


	#Generate graph, edge weights
	file = open("Data/Sacramentorealestatetransactions.csv", "rU")
	file.readline() #ignore first line
	G1 = TUNGraph.New()
	locations = TIntFltPrH()
	dataset = TIntIntVH()
	counter = 0
	for line in file:
		G1.AddNode(counter)
		temp = TFltPr(float(line.split(",")[10]),float(line.split(",")[11]))
		locations.AddDat(counter, temp)
		tempData = TIntV()
		tempData.Add(int(line.split(",")[4]))
		tempData.Add(int(line.split(",")[5]))
		tempData.Add(int(line.split(",")[6]))
		if(line.split(",")[7] == "Residential"):
			tempData.Add(1)
		elif(line.split(",")[7] == "Condo"):
			tempData.Add(2)
		elif(line.split(",")[7] == "Multi-Family"):
			tempData.Add(3)
		else:
			tempData.Add(4)
		tempData.Add(int(line.split(",")[9]))
		dataset.AddDat(counter, tempData)
		counter = counter + 1

	#Remove random subset of nodes for test and validation sets
	testList = TIntV()
	for i in range(testSetSize):
		temp = G1.GetRndNId()
		G1.DelNode(temp)
		testList.Add(temp)

	validationList = TIntV()
	for i in range(validationSetSize):
		temp = G1.GetRndNId()
		G1.DelNode(temp)
		validationList.Add(temp)

	#For each node, find closest neightbors and add edge, weight = 5/distance
	edgeWeights = TIntPrFltH()
	for NI in G1.Nodes():
		distances = TIntFltH()
		lat1 = locations.GetDat(NI.GetId()).GetVal1()
		lon1 = locations.GetDat(NI.GetId()).GetVal2()
		for NI2 in G1.Nodes():
			if(NI.GetId() != NI2.GetId()):
				lat2 = locations.GetDat(NI2.GetId()).GetVal1()
				lon2 = locations.GetDat(NI2.GetId()).GetVal2()
				dlon = math.radians(lon2 - lon1)
				dlat = math.radians(lat2 - lat1)
				a2 = math.pow(math.sin(dlat/2),2) + math.cos(lat1)*math.cos(lat2) * math.pow(math.sin(dlon/2),2)
				c = 2 * math.atan2( math.sqrt(a2), math.sqrt(1-a2) ) 
				dist = 3961 * c
				distances.AddDat(NI2.GetId(), dist)

		distances.Sort(False, True)
		it = distances.BegI()
		for j in range(numNeighs):
			if (not G1.IsEdge(NI.GetId(), it.GetKey())):
				G1.AddEdge(NI.GetId(), it.GetKey())
				#Add edge weight
				temp = TIntPr(min(NI.GetId(), it.GetKey()), max(NI.GetId(), it.GetKey()))
				edgeWeights.AddDat(temp, 1/(it.GetDat()+ 0.1))
			it.Next()		

	nodes = G1.GetNodes()
	edges = G1.GetEdges()
	print nodes, edges
	print GetBfsFullDiam(G1, 1000, False);

	#Get side information
	a = np.zeros((sizeData, nodes))
	counter = 0
	for NI in G1.Nodes():
		a[0,counter] = dataset.GetDat(NI.GetId())[0]
		a[1,counter] = dataset.GetDat(NI.GetId())[1]
		a[2,counter] = dataset.GetDat(NI.GetId())[2]
		a[3,counter] = dataset.GetDat(NI.GetId())[3]
		a[4,counter] = dataset.GetDat(NI.GetId())[4]
		counter = counter + 1

	#Initialize variables to 0
	x = np.zeros((sizeOptVar,nodes))
	u = np.zeros((sizeOptVar,2*G1.GetEdges()))
	z = np.zeros((sizeOptVar,2*G1.GetEdges()))

	#Run regularization path
	[plot1, plot2] = [TFltV(), TFltV()]
	while(lamb <= thresh):
		(x, u, z, pl1, pl2) = runADMM(G1, sizeOptVar, sizeData, lamb, rho + math.sqrt(lamb), numiters, x, u ,z, a, edgeWeights, useConvex, epsilon)
		print "Lambda = ", lamb
		mse = 0
		#Calculate accuracy on test set
		for i in testList:
			#Find closest neighbors
			distances = TIntFltH()
			lat1 = locations.GetDat(i).GetVal1()
			lon1 = locations.GetDat(i).GetVal2()
			counter = 0
			for NI2 in G1.Nodes():
				lat2 = locations.GetDat(NI2.GetId()).GetVal1()
				lon2 = locations.GetDat(NI2.GetId()).GetVal2()
				dlon = math.radians(lon2 - lon1)
				dlat = math.radians(lat2 - lat1)
				a2 = math.pow(math.sin(dlat/2),2) + math.cos(lat1)*math.cos(lat2) * math.pow(math.sin(dlon/2),2)
				c = 2 * math.atan2( math.sqrt(a2), math.sqrt(1-a2) ) 
				dist = 3961 * c
				distances.AddDat(counter, dist)		
				counter = counter + 1
			distances.Sort(False, True)

			#Predict price
			xpred = Variable(sizeOptVar,1)
			g = 0
			it = distances.BegI()
			for j in range(numNewNeighs):
				#neighsList.AddDat(it.GetKey(), it.GetDat())
				weight = 1/(it.GetDat()+ 0.1)
				g = g + weight*norm(xpred - x[0, it.GetKey()])
				it.Next()	
		
			objective = Minimize(g)
			constraints = []
			p = Problem(objective, constraints)
			result = p.solve()	

			#Find MSE
			#print xpred.value[0], float(dataset.GetDat(i)[4])/100000
			mse = mse + math.pow(xpred.value[0] - float(dataset.GetDat(i)[4])/100000,2)/testSetSize
		print mse, "= mse"
		plot1.Add(lamb)
		plot2.Add(mse)
		lamb = lamb + updateVal


	#Print/Save plot
	pl1 = np.array(plot1)
	pl2 = np.array(plot2)
	plt.plot(pl1, pl2)
	plt.savefig('image_housing',bbox_inches='tight')



if __name__ == '__main__':
	main()
