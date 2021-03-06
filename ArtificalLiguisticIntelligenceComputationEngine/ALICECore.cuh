#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <iostream>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <npp.h>//CUDA Min/Max data values

#include <stdio.h>
#include <algorithm>
#include<iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <iterator>
#include <vector>
#include <time.h>
#include "LoadSaveNeuralNetwork.h"

#define FRAMES_PER_BUFFER (2205)
#define NUM_OF_TIME_VARIABLES (5)

using namespace std;

__device__ float Sigmoid(float a);

__global__ void CoreAIKernel(
	/*
	[0]NumOfCycles
	[1]NumOfInputs
	[2]NumOfOutputs
	-----
	[3 + (InputNum * 3)]NumOfElements
	[4 + (InputNum * 3)]InputToBufferNodeID
	[5 + (InputNum * 3)]IncrementElementEveryXNumOfCylce (Exception: when 0, no incrementation)
	----
	[3 + (NeuralNetworkSetup[1] * 3) + (OutputNum * 3)]NumOfElements
	[4 + (NeuralNetworkSetup[1] * 3) + (OutputNum * 3)]OutputFromBufferNodeID
	[5 + (NeuralNetworkSetup[1] * 3) + (OutputNum * 3)] IncrementElementEveryXNumOfCylce (Exception: when 0, no incrementation)
	----

	*/
	unsigned int *NeuralNetworkSetup,
	float *InputData,
	float *OutputData,
	float *BufferNodeValues,
	unsigned int NumOfNeurons,
	/*
	0 = Sum ((input[] * weight[]) + bias[])
	1 = Sigmoid (Sum ((input[] * weight[]) + bias[]))
	2 = Tanh (Sum ((input[] * weight[]) + bias[]))
	3 = Sum
	4 = Multiply
	5 = Min
	6 = Max
	*/
	unsigned int *TypeOfNode,
	unsigned int *NeuronNodeStartBuffer,
	unsigned int *NeuronNodeLength,
	unsigned int *NeuronInputNodeIDsBuffer,
	unsigned int *NeuronOutputNodeIDsBuffer,
	float *NeuronOutputNodeBuffer,
	float *NeuronWeightsBuffer,
	float *NeuronBiasBuffer,
	float *ScoringVariables,
	float *NetworkScore
);

cudaError_t RunAICoreOnGPU(
	unsigned int NeuralNetworkSetupSize,
	unsigned int *NeuralNetworkSetup,
	unsigned int InputDataSize,
	float *InputData,
	unsigned int OutputDataSize,
	float *OutputData,
	unsigned int NumOfBufferNodes,
	float *BufferNodeValues,
	unsigned int NumOfNeurons,
	unsigned int NumOfNeuronInputs,
	unsigned int *TypeOfNode,
	unsigned int *NeuronNodeStartBuffer,
	unsigned int *NeuronNodeLength,
	unsigned int *NeuronInputNodeIDsBuffer,
	unsigned int *NeuronOutputNodeIDsBuffer,
	float *NeuronOutputNodeBuffer,
	unsigned int NumOfNeuronWeights,
	float *NeuronWeightsBuffer,
	float *NeuronBiasBuffer,
	unsigned int NumOfScoringVariables,
	float *ScoringVariables,
	float *NetworkScore
);

/*
int RunAICore(
	NeuralNetworkModule& NeuralNetworkIndividual,
	vector<float>& InputData,
	vector<float>& OutputData,
	vector<float>& ScoringVariables,
	float& NetworkScore
);*/












__device__ void RunLSTMUnitStageOneForgetGate(
	float InputValue,						//xt
	float PreviousOutputValue,				//ht-1
	//Weights
	float ForgetGateInputWeight,			//wf
	float ForgetGatePreviousInputWeight,	//uf
	//End Weights
	//Biases
	float FogetGateBias,
	//End Biases
	float &ForgetGateValue);					//ft

__device__ void RunLSTMUnitStageOneInputGate(
	float InputValue,						//xt
	float PreviousOutputValue,				//ht-1
	//Weights
	float InputGateInputWeight,				//wi
	float InputGatePreviousInputWeight,		//ui
	//End Weights
	//Biases
	float InputGateBias,
	//End Biases
	float &InputGateValue);					//it

__device__ void RunLSTMUnitStageOneCandidateGate(
	float InputValue,						//xt
	float PreviousOutputValue,				//ht-1
	//Weights
	float CandidateGateInputWeight,			//wc
	float CandidateGatePreviousInputWeight,	//uc
	//End Weights
	//Biases
	float CandidateGateBias,
	//End Biases
	float &CandidateValue);			//~ct

__device__ void RunLSTMUnitStageTwoOutputGate(
	float InputValue,						//xt
	float PreviousOutputValue,				//ht-1
	//Weights
	float OutputGateInputWeight,			//wo
	float OutputGatePreviousInputWeight,	//uo
	//End Weights
	//Biases
	float OutputGateBias,
	//End Biases
	float &OutputGateValue					//ot
);

__device__ void RunLSTMUnitStageTwoCellState(
	float PreviousCellStateValue,	//ct-1
	float ForgetGateValue,			//ft
	float InputGateValue,			//it
	float CandidateValue,			//~ct
	float &CellStateValue			//ct
);

__device__ void RunLSTMUnitStageThree(
	float OutputGateValue,			//ot
	float CellStateValue,			//ct
	float &OutputValue);				//ht


