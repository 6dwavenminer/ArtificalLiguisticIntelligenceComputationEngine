
#include "NeuralNetworkModuleIntergrationAndSeperation.h" 

int IntergrateUnsignedIntVectorNeuron(
	vector<unsigned int> *InputVector,
	vector<unsigned int> *VectorToIntergrate,
	vector<unsigned int>& OutputVector
) {
	vector<unsigned int> BufferOutputVector;
	while (BufferOutputVector.size() != 0) {
		BufferOutputVector.clear();
	}
	if ((*InputVector).size() != 0) {
		BufferOutputVector.resize((*InputVector).size());
		copy((*InputVector).begin(), (*InputVector).end(), BufferOutputVector.begin());
	}
	for (unsigned int i = 0; i < (*VectorToIntergrate).size(); i++) {
		BufferOutputVector.push_back((*VectorToIntergrate)[i] + (*InputVector).size());
	}
	OutputVector.resize(BufferOutputVector.size());
	OutputVector.insert(OutputVector.begin(), BufferOutputVector.begin(), BufferOutputVector.end());
	return 1;
}

int IntergrateUnsignedFloatVectorNeuron(
	vector<float> *InputVector,
	vector<float> *VectorToIntergrate,
	vector<float>& OutputVector
) {
	vector<unsigned int> BufferOutputVector;
	while (BufferOutputVector.size() != 0) {
		BufferOutputVector.clear();
	}
	if ((*InputVector).size() != 0) {
		BufferOutputVector.resize((*InputVector).size());
		copy((*InputVector).begin(), (*InputVector).end(), BufferOutputVector.begin());
	}
	for (unsigned int i = 0; i < (*VectorToIntergrate).size(); i++) {
		BufferOutputVector.push_back((*VectorToIntergrate)[i] + (*InputVector).size());
	}
	OutputVector.resize(BufferOutputVector.size());
	OutputVector.insert(OutputVector.begin(), BufferOutputVector.begin(), BufferOutputVector.end());
	return 1;
}



int SeperateLSTMFloat(
	vector<float> *InputVector,
	vector<float>& OutputVector,
	vector<float>& SeperationOutputVector,
	unsigned int Offset
) {
	vector<float> BufferOutputVector;
	while (BufferOutputVector.size() != 0) {
		BufferOutputVector.clear();
	}
	BufferOutputVector.resize(Offset);
	copy((*InputVector).begin(), (*InputVector).begin() + Offset, BufferOutputVector.begin());

	while (SeperationOutputVector.size() != 0) {
		SeperationOutputVector.clear();
	}
	OutputVector.resize(BufferOutputVector.size());
	OutputVector.insert(OutputVector.begin(), BufferOutputVector.begin(), BufferOutputVector.end());
	SeperationOutputVector.resize((*InputVector).size() - Offset);
	copy((*InputVector).begin() + Offset, (*InputVector).end(), SeperationOutputVector.begin());
	return 1;
}

int SeperateLSTMUInt(
	vector<unsigned int> *InputVector,
	vector<unsigned int>& OutputVector,
	vector<unsigned int>& SeperationOutputVector,
	unsigned int Offset
) {
	vector<unsigned int> BufferOutputVector;
	while (BufferOutputVector.size() != 0) {
		BufferOutputVector.clear();
	}
	BufferOutputVector.resize(Offset);
	copy((*InputVector).begin(), (*InputVector).begin() + Offset, BufferOutputVector.begin());
	
	while (SeperationOutputVector.size() != 0) {
		SeperationOutputVector.clear();
	}
	OutputVector.resize(BufferOutputVector.size());
	OutputVector.insert(OutputVector.begin(), BufferOutputVector.begin(), BufferOutputVector.end());
	SeperationOutputVector.resize((*InputVector).size() - Offset);
	copy((*InputVector).begin() + Offset, (*InputVector).end(), SeperationOutputVector.begin());
	return 1;
}



