import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil, factorial, pi
from cmath import exp,log
import os
#import pandas as pd
import cvxopt
import time
data_path = ''.join(map(lambda string: string+'\\', os.path.realpath(__file__).split('\\')[:-1]))+'data/'

from VisualizationTools import PopEV_Y_maker, getMNIST_data_full
from Algorithms import GLSS_estimator, CurveDiscoveryHSpace, CurveDiscoveryNuSpace, IntegrationNodes, Vectorwise_StieltjesTransformEstimator


def EstimateDotProduct_LedoitWolf(SampEV,PopEV_est,c,SampVecNr=1,PopVecNr=1):
	d = len(SampEV)
	#SampEV.sort()
	if SampVecNr==1:
		dist = (SampEV[-SampVecNr]-SampEV[-SampVecNr-1])/2
		f_curve = (SampEV[-SampVecNr]-dist, dist , SampEV[-SampVecNr]+dist)
	else:
		f_curve = ((SampEV[-SampVecNr]+SampEV[-SampVecNr-1])/2, (SampEV[-SampVecNr+1]-SampEV[-SampVecNr])/2 , (SampEV[-SampVecNr]+SampEV[-SampVecNr+1])/2)
	N = 20
	nodes,weights = IntegrationNodes(N,f_curve[0],f_curve[1],f_curve[2])
	lam = PopEV_est[-PopVecNr]
	ulSampStil = c*np.sum(1/(SampEV[np.newaxis,:]-nodes[:,np.newaxis]), axis=1)/d - (1-c)/nodes
	approximatingTrace = -1/nodes*1/(1+ulSampStil*lam)
	DotProductEstimator = -1/(2*pi*1j)*np.sum(weights*approximatingTrace-np.conj(weights*approximatingTrace))
	return DotProductEstimator

def EstimateDotProduct_MPI(SampEV,c,SampVecNr=1,PopVecNr=1,tau=0.01,kappa=np.inf,resolution=10**(-2)):
	#SampEV.sort()
	d = len(SampEV)
	g_curves = CurveDiscoveryHSpace(SampEV,c,tau=tau,kappa=kappa,resolution=resolution)
	f_curves = CurveDiscoveryNuSpace(SampEV,c,tau=tau,kappa=kappa,g_curves=g_curves)

	g_curve = g_curves[-PopVecNr]
	f_curve = f_curves[-SampVecNr]

	z = np.array([g_curve[0],g_curve[2]])+0.00001j
	s = Vectorwise_StieltjesTransformEstimator(z,SampEV,c,tau=tau,kappa=kappa)
	Phi = np.real((1-c*z*s-c)*z)

	#ensure the eigenvalues are sufficiently well separated
	if not (SampEV[-PopVecNr-1] < Phi[0] and Phi[0] < SampEV[-PopVecNr] and SampEV[-PopVecNr] < Phi[1]):
		print("The population eigenvalues are not sufficiently well separated!")
		return np.nan
	if not (SampEV[-SampVecNr-1] < f_curve[0] and f_curve[0] < SampEV[-SampVecNr] and SampEV[-SampVecNr] < f_curve[2]):
		print("The population eigenvalues are not sufficiently well separated!")
		return np.nan

	DotProductEstimator = d*GLSS_estimator(SampEV,c,lambda z : 1,lambda z : 1,[g_curve],[f_curve],tau=tau,kappa=kappa)
	return DotProductEstimator



if __name__ == "__main__":
	resolution=1/100
	ToCompare = {'LedoitWolf', 'MPI'}
	###################
	if 1: #Figure 7c
		d = 784
		tau,kappa = (0.05,25)
		ExampleNumber=3
		SampVecNr = 1
		PopVecNr = 1
	###################

	

	
	try:
		n_List = np.load(data_path+'Fig8_n_List.npy')
		AvgTimes_LedoitWolf = np.load(data_path+'Fig8_AvgTimes_LedoitWolf_Ex{}.npy'.format(ExampleNumber))
		LedoitWolf_EstimatedEV_List = []
		AllDataMatrices = []
		AllPopEVs = []
		AllChosenDims = []
		for n in n_List:
			AllDataMatrices.append(np.load(data_path+'Fig8_DataMatrices_n={}_Ex{}.npy'.format(int(n),ExampleNumber)))
			AllPopEVs.append(np.load(data_path+'Fig8_PopEVs_n={}_Ex{}.npy'.format(int(n),ExampleNumber)))
			AllChosenDims.append(np.load(data_path+'Fig8_DimsChosen_n={}_Ex{}.npy'.format(int(n),ExampleNumber)))
			LedoitWolf_EstimatedEV_List.append(np.load(data_path+'Fig8_LedoitWolf_Estimators_n={}_Ex{}.npy'.format(int(n),ExampleNumber)))
			_,NN = LedoitWolf_EstimatedEV_List[0].shape
	except:
		print('#############################')
		print("Error: could not load the results of the Ledoit-Wolf estimator")
		print("try running Fig_8_1_Preparation.R first")
		print('#############################')
		assert 0==1

	AvgErrors = dict()
	Variances = dict()
	AvgTimes = dict()
	AllErrors = dict()
	
	for key in ToCompare:
		AvgErrors[key] = []
		Variances[key] = []
		AvgTimes[key] = []
		AllErrors[key] = []

	MNIST_data = getMNIST_data_full()

	for k in range(len(n_List)):
		n=int(n_List[k])
		print('n=',n)

		TimeDiff = dict()
		Errors = dict()

		for key in ToCompare:
			TimeDiff[key] = 0
			Errors[key] = []

		LedoitWolf_EstimatedEVs = LedoitWolf_EstimatedEV_List[k]

		DataMatrices = AllDataMatrices[k]
		PopEVs = AllPopEVs[k]
		ChosenDims = AllChosenDims[k]

		for i in range(NN):
			print(i)
			Y = DataMatrices[i,:,:]

			S = Y@Y.T/n
			SampEV,SampVec = np.linalg.eigh(S)
			u = SampVec[:,-SampVecNr]
			c = d/n

			Y_full = MNIST_data[(ChosenDims[i,:]-1).astype(int),:]
			Sigma = Y_full@Y_full.T/(Y_full.shape[1])
			PopEV, PopVec = np.linalg.eigh(Sigma)
			v = PopVec[:,-PopVecNr]


			trueDotProduct = np.abs(u.T@v)**2

			print('true:',trueDotProduct)


			if 'LedoitWolf' in ToCompare:
				LedoitWolf_est = EstimateDotProduct_LedoitWolf(SampEV,LedoitWolf_EstimatedEVs[:,i],c,SampVecNr=SampVecNr,PopVecNr=PopVecNr)
				print('LW:',LedoitWolf_est)
				Errors['LedoitWolf'].append(LedoitWolf_est-trueDotProduct)

			if 'MPI' in ToCompare:
				start = time.time()
				Est = EstimateDotProduct_MPI(SampEV,c,SampVecNr=SampVecNr,PopVecNr=PopVecNr,tau=tau,kappa=kappa,resolution=resolution)
				end = time.time()
				print('MPI:',Est)
				Errors['MPI'].append(Est-trueDotProduct)
				TimeDiff['MPI'] += end - start

		for key in ToCompare:
			AvgTimes[key].append(TimeDiff[key]/NN)
			AvgErrors[key].append(sum(np.abs(Errors[key]))/NN)
			mean = sum(Errors[key])/NN
			Variances[key].append(sum(np.abs(np.array(Errors[key])-mean)**2)/(NN-1))
			AllErrors[key].append(np.abs(Errors[key]))

	if 'LedoitWolf' in ToCompare:
		AvgTimes['LedoitWolf'] = AvgTimes_LedoitWolf

	for key in ToCompare:
		np.save(data_path+'Fig8_AvgTimes_vec{}{}_{}_Ex{}'.format(SampVecNr,PopVecNr,key,ExampleNumber),np.array(AvgTimes[key]))
		np.save(data_path+'Fig8_AvgErrors_vec{}{}_{}_Ex{}'.format(SampVecNr,PopVecNr,key,ExampleNumber),np.array(AvgErrors[key]))
		np.save(data_path+'Fig8_Variances_vec{}{}_{}_Ex{}'.format(SampVecNr,PopVecNr,key,ExampleNumber),np.array(Variances[key]))
		np.save(data_path+'Fig8_AllErrors_vec{}{}_{}_Ex{}'.format(SampVecNr,PopVecNr,key,ExampleNumber),np.array(AllErrors[key]))
