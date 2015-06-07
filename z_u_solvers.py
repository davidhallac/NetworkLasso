import numpy as np
from numpy import linalg as LA
import math

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

		if(rho*math.pow(d+epsilon,2) - 8*c >= 0):
			theta1 = (rho*(d+epsilon) + math.sqrt(math.pow(rho,2)*math.pow(d+epsilon,2) - 8*rho*c)) / (4*rho*d+0.00000001)
			theta1 = min(max(theta1,0),0.5)
			phi = math.log(1 + d*(1-2*theta1)/epsilon)
			objective1 = c*phi + rho*math.pow(d,2)*(math.pow(theta1,2))

			theta2 = (rho*(d+epsilon) - math.sqrt(math.pow(rho,2)*math.pow(d+epsilon,2) - 8*rho*c)) / (4*rho*d+0.00000001)
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