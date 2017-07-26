#pragma once


#include <vector>
#include "LoadSaveNeuralNetwork.h"
#define NumofNNInputs 6

using namespace std;

int IntergrateUnsignedIntVectorNeuron(
	vector<unsigned int> *InputVector,
	vector<unsigned int> *VectorToIntergrate,
	vector<unsigned int>& OutputVector
);

int IntergrateUnsignedFloatVectorNeuron(
	vector<float> *InputVector,
	vector<float> *VectorToIntergrate,
	vector<float>& OutputVector
);

//IntergrateNeuralNetworkLSTMPlusBufferNodes(
//&inputs
//outputs
//)


int IntergrateNeuralNetworkModule(
	OLD_NeuralNetworkModule *InputModule,
	OLD_NeuralNetworkModule *ModuleToIntergrate,
	NeuralNetworkModule& OutputModule,

	unsigned int NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int &BufferNodeOffset,
	unsigned int &LSTMOffset,
	unsigned int& NeuronOffset,
	unsigned int& NeuronWeightSize
);


int SeperateLSTMFloat(
	vector<float> *InputVector,
	vector<float>& OutputVector,
	vector<float>& SeperationOutputVector,
	unsigned int Offset
);

int SeperateLSTMUInt(
	vector<unsigned int> *InputVector,
	vector<unsigned int>& OutputVector,
	vector<unsigned int>& SeperationOutputVector,
	unsigned int Offset
);

int SeperateNeuralNetworkModule(
	OLD_NeuralNetworkModule *InputModule,
	OLD_NeuralNetworkModule &BaseOutputModule,
	OLD_NeuralNetworkModule &SeperatedOutputModule,

	unsigned int NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int BufferNodeOffset,
	unsigned int LSTMOffset,
	unsigned int NeuronOffset,
	unsigned int NeuronWeightSize
);


