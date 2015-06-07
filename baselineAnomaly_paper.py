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




def main():

	eventThresh = 1e-51
	numTrials = 300
	multUpdate = 1.5

	results = np.zeros((numTrials,6))
	threshSched = np.zeros((numTrials,1))

	#Generate graph, edge weights
	file = open("Data/CalIt2.csv", "rU")
	dataset = TIntFltVH()
	G1 = TUNGraph.New()
	counter = 0
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
		

	nodes = G1.GetNodes()
	edges = G1.GetEdges()

	#Save side information
	a = np.zeros((2, nodes))
	for NI in G1.Nodes():
		a[0,NI.GetId()] = dataset.GetDat(NI.GetId())[0]
		a[1,NI.GetId()] = dataset.GetDat(NI.GetId())[1]
	
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
		baseline[0,i] = np.mean(temp)
		baseline[1,i] = np.mean(temp)

	x = np.zeros((nodes,1))
	for i in range(nodes):
		x[i,0] = ((math.exp(-1*baseline[0,i  % (48*7)]) * math.pow(baseline[0,i  % (48*7)], a[0,i])) / math.factorial(a[0,i])) * \
			((math.exp(-1*baseline[1,i  % (48*7)]) * math.pow(baseline[1,i  % (48*7)], a[1,i])) / math.factorial(a[1,i]))

	#Compare to actual events
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






	for z in range(numTrials):

		#Predict events
		counter = 0
		maxLength = 0
		for i in range(nodes-1):
			if(x[i,0] < eventThresh and ((a[0,i] > baseline[0,i  % (48*7)]) or (a[1,i] > baseline[1,i  % (48*7)])) and \
				not (x[i-1,0] < eventThresh and ((a[0,i-1] > baseline[0,(i-1) % (48*7)]) or (a[1,i-1] > baseline[1,(i-1)  % (48*7)])))):
				beginning = i
			if(x[i,0] < eventThresh and ((a[0,i] > baseline[0,i  % (48*7)]) or (a[1,i] > baseline[1,i  % (48*7)])) and \
				not (x[i+1,0] < eventThresh and ((a[0,i+1] > baseline[0,(i+1) % (48*7)]) or (a[1,i+1] > baseline[1,(i+1)  % (48*7)])))):
				end  = i
				#print "Event ", counter, " starts at ", beginning, "and is length ", i - beginning + 1
				maxLength = max(maxLength, i-beginning)
				counter = counter + 1
		#print maxLength, " = maximum length"
		#print counter, " events predicted"

		numPred = counter

		counter = 0
		start = 0
		correct = 0
		for i in range(nodes):		

			if(x[i,0] < eventThresh and ((a[0,i] > baseline[0,i  % (48*7)]) or (a[1,i] > baseline[1,i  % (48*7)]))):
				counter = counter + 1
				#Check if it was correctly counted
				if(sum(truth[0,i-1:i+1]) > 0):
					correct = correct + 1

		#print counter, " timestamps triggered"
		#print correct, " correct answers"

		timestampsPred = counter
		timestampsCorr = correct


		#See how many of the 30 events were detected
		numevents = 0
		for meeting in events:
			start = meeting.GetVal1()
			end = meeting.GetVal2()
			counter = start
			while (counter <= end):
				if(x[counter,0] < eventThresh and ((a[0,counter] > baseline[0,counter  % (48*7)]) or (a[1,counter] > baseline[1,counter  % (48*7)]))):
					numevents = numevents + 1
					break
				counter = counter + 1
				if (counter > end):
					1+1
					#print "Missed event from ", start, " to ", end

		#print numevents, " events detected"


		results[z,:] = [eventThresh, numevents, numPred, timestampsCorr, timestampsPred, maxLength]
		threshSched[z,0] = eventThresh

		#Update eventThresh
		eventThresh = multUpdate*eventThresh


	np.set_printoptions(formatter={'float': '{: 0.7f}'.format})
	print(threshSched)
	np.set_printoptions(suppress=True)
	print results

	np.savetxt("results_baseline.csv", results, delimiter=",")

if __name__ == '__main__':
	main()
