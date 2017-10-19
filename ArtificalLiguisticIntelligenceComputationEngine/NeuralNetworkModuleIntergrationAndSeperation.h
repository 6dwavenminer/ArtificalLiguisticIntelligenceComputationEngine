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



