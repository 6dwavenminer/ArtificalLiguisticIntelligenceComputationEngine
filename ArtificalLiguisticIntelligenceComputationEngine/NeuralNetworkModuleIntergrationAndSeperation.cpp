
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

//IntergrateNeuralNetworkLSTMPlusBufferNodes(
//&inputs
//outputs
//)


int IntergrateNeuralNetworkModule(
	OLD_NeuralNetworkModule *InputModule,
	OLD_NeuralNetworkModule *ModuleToIntergrate,
	OLD_NeuralNetworkModule& OutputModule,

	unsigned int NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int &BufferNodeOffset,
	unsigned int &LSTMOffset,
	unsigned int& NeuronOffset,
	unsigned int& NeuronWeightSize
) {
	vector<unsigned int> IntergrationLSTMInputNodeIDBuffer;
	vector<unsigned int> IntergrationLSTMOutputNodeIDBuffer;

	if (NetworkModule == 1) {
		BufferNodeOffset = (*InputModule).BufferNodeValues.size() - 11;//Change to - num of prev outputs
	}else {
		BufferNodeOffset = (*InputModule).BufferNodeValues.size();
	}
	LSTMOffset = (*InputModule).LSTMInputNodeIDBuffer.size();
	NeuronOffset = (*InputModule).NeuronNodeStartBuffer.size();
	NeuronWeightSize = (*InputModule).NeuronWeightsBuffer.size();

	OutputModule.BufferNodeValues.clear();
	if ((*InputModule).BufferNodeValues.size() != 0) {
		OutputModule.BufferNodeValues.insert(OutputModule.BufferNodeValues.begin(), (*InputModule).BufferNodeValues.begin(), (*InputModule).BufferNodeValues.end());
	}
	if ((*ModuleToIntergrate).BufferNodeValues.size() != 0) {
		OutputModule.BufferNodeValues.insert(OutputModule.BufferNodeValues.end(), (*ModuleToIntergrate).BufferNodeValues.begin(), (*ModuleToIntergrate).BufferNodeValues.end());
	}

	//LSTMInputNodeIDBuffer
	if ((*InputModule).LSTMInputNodeIDBuffer.size() != 0) {
		IntergrationLSTMInputNodeIDBuffer.resize((*InputModule).LSTMInputNodeIDBuffer.size());
		copy((*InputModule).LSTMInputNodeIDBuffer.begin(), (*InputModule).LSTMInputNodeIDBuffer.end(), IntergrationLSTMInputNodeIDBuffer.begin());
	}
	if ((*ModuleToIntergrate).LSTMInputNodeIDBuffer.size() != 0) {
		for (unsigned int i = 0; i < (*ModuleToIntergrate).LSTMInputNodeIDBuffer.size(); i++) {
			//
			if ((NetworkModule == 1) && ((*ModuleToIntergrate).LSTMInputNodeIDBuffer[i] <= NumofNNInputs)) {
				IntergrationLSTMInputNodeIDBuffer.push_back((*ModuleToIntergrate).LSTMInputNodeIDBuffer[i]);
			}else {
				IntergrationLSTMInputNodeIDBuffer.push_back((*ModuleToIntergrate).LSTMInputNodeIDBuffer[i] + BufferNodeOffset);
			}
		}
	}
	OutputModule.LSTMInputNodeIDBuffer.resize(IntergrationLSTMInputNodeIDBuffer.size());
	copy(IntergrationLSTMInputNodeIDBuffer.begin(), IntergrationLSTMInputNodeIDBuffer.end(), OutputModule.LSTMInputNodeIDBuffer.begin());

	//LSTMOutputNodeIDBuffer
	if ((*InputModule).LSTMOutputNodeIDBuffer.size() != 0) {
		IntergrationLSTMOutputNodeIDBuffer.resize((*InputModule).LSTMOutputNodeIDBuffer.size());
		copy((*InputModule).LSTMOutputNodeIDBuffer.begin(), (*InputModule).LSTMOutputNodeIDBuffer.end(), IntergrationLSTMOutputNodeIDBuffer.begin());
	}
	if ((*ModuleToIntergrate).LSTMOutputNodeIDBuffer.size() != 0) {
		for (unsigned int i = 0; i < (*ModuleToIntergrate).LSTMOutputNodeIDBuffer.size(); i++) {
			IntergrationLSTMOutputNodeIDBuffer.push_back((*ModuleToIntergrate).LSTMOutputNodeIDBuffer[i] + BufferNodeOffset);
		}
	}
	OutputModule.LSTMOutputNodeIDBuffer.resize((*ModuleToIntergrate).LSTMOutputNodeIDBuffer.size());
	copy(IntergrationLSTMOutputNodeIDBuffer.begin(), IntergrationLSTMOutputNodeIDBuffer.end(), OutputModule.LSTMOutputNodeIDBuffer.begin());

	OutputModule.LSTMWeightsListBuffer.clear();
	if ((*InputModule).LSTMWeightsListBuffer.size() != 0) {
		OutputModule.LSTMWeightsListBuffer.insert(OutputModule.LSTMWeightsListBuffer.begin(), (*InputModule).LSTMWeightsListBuffer.begin(), (*InputModule).LSTMWeightsListBuffer.end());
	}
	if ((*ModuleToIntergrate).LSTMWeightsListBuffer.size() != 0) {
		OutputModule.LSTMWeightsListBuffer.insert(OutputModule.LSTMWeightsListBuffer.end(), (*ModuleToIntergrate).LSTMWeightsListBuffer.begin(), (*ModuleToIntergrate).LSTMWeightsListBuffer.end());
	}
	//  
	OutputModule.LSTMPreviousCellStateValueBuffer.clear();
	if ((*InputModule).LSTMPreviousCellStateValueBuffer.size() != 0) {
		OutputModule.LSTMPreviousCellStateValueBuffer.insert(OutputModule.LSTMPreviousCellStateValueBuffer.begin(), (*InputModule).LSTMPreviousCellStateValueBuffer.begin(), (*InputModule).LSTMPreviousCellStateValueBuffer.end());
	}
	if ((*ModuleToIntergrate).LSTMPreviousCellStateValueBuffer.size() != 0) {
		OutputModule.LSTMPreviousCellStateValueBuffer.insert(OutputModule.LSTMPreviousCellStateValueBuffer.end(), (*ModuleToIntergrate).LSTMPreviousCellStateValueBuffer.begin(), (*ModuleToIntergrate).LSTMPreviousCellStateValueBuffer.end());
	}

	OutputModule.LSTMPreviousOutputValueBuffer.clear();
	if ((*InputModule).LSTMPreviousOutputValueBuffer.size() != 0) {
		OutputModule.LSTMPreviousOutputValueBuffer.insert(OutputModule.LSTMPreviousOutputValueBuffer.begin(), (*InputModule).LSTMPreviousOutputValueBuffer.begin(), (*InputModule).LSTMPreviousOutputValueBuffer.end());
	}
	if ((*ModuleToIntergrate).LSTMPreviousOutputValueBuffer.size() != 0) {
		OutputModule.LSTMPreviousOutputValueBuffer.insert(OutputModule.LSTMPreviousOutputValueBuffer.end(), (*ModuleToIntergrate).LSTMPreviousOutputValueBuffer.begin(), (*ModuleToIntergrate).LSTMPreviousOutputValueBuffer.end());
	}

	OutputModule.LSTMForgetGateValueBuffer.clear();
	if ((*InputModule).LSTMForgetGateValueBuffer.size() != 0) {
		OutputModule.LSTMForgetGateValueBuffer.insert(OutputModule.LSTMForgetGateValueBuffer.begin(), (*InputModule).LSTMForgetGateValueBuffer.begin(), (*InputModule).LSTMForgetGateValueBuffer.end());
	}
	if ((*ModuleToIntergrate).LSTMForgetGateValueBuffer.size() != 0) {
		OutputModule.LSTMForgetGateValueBuffer.insert(OutputModule.LSTMForgetGateValueBuffer.end(), (*ModuleToIntergrate).LSTMForgetGateValueBuffer.begin(), (*ModuleToIntergrate).LSTMForgetGateValueBuffer.end());
	}

	OutputModule.LSTMInputGateValueBuffer.clear();
	if ((*InputModule).LSTMInputGateValueBuffer.size() != 0) {
		OutputModule.LSTMInputGateValueBuffer.insert(OutputModule.LSTMInputGateValueBuffer.begin(), (*InputModule).LSTMInputGateValueBuffer.begin(), (*InputModule).LSTMInputGateValueBuffer.end());
	}
	if ((*ModuleToIntergrate).LSTMInputGateValueBuffer.size() != 0) {
		OutputModule.LSTMInputGateValueBuffer.insert(OutputModule.LSTMInputGateValueBuffer.end(), (*ModuleToIntergrate).LSTMInputGateValueBuffer.begin(), (*ModuleToIntergrate).LSTMInputGateValueBuffer.end());
	}

	OutputModule.LSTMCandidateValueBuffer.clear();
	if ((*InputModule).LSTMCandidateValueBuffer.size() != 0) {
		OutputModule.LSTMCandidateValueBuffer.insert(OutputModule.LSTMCandidateValueBuffer.begin(), (*InputModule).LSTMCandidateValueBuffer.begin(), (*InputModule).LSTMCandidateValueBuffer.end());
	}
	if ((*ModuleToIntergrate).LSTMCandidateValueBuffer.size() != 0) {
		OutputModule.LSTMCandidateValueBuffer.insert(OutputModule.LSTMCandidateValueBuffer.end(), (*ModuleToIntergrate).LSTMCandidateValueBuffer.begin(), (*ModuleToIntergrate).LSTMCandidateValueBuffer.end());
	}

	OutputModule.LSTMOutputGateValueBuffer.clear();
	if ((*InputModule).LSTMOutputGateValueBuffer.size() != 0) {
		OutputModule.LSTMOutputGateValueBuffer.insert(OutputModule.LSTMOutputGateValueBuffer.begin(), (*InputModule).LSTMOutputGateValueBuffer.begin(), (*InputModule).LSTMOutputGateValueBuffer.end());
	}
	if ((*ModuleToIntergrate).LSTMOutputGateValueBuffer.size() != 0) {
		OutputModule.LSTMOutputGateValueBuffer.insert(OutputModule.LSTMOutputGateValueBuffer.end(), (*ModuleToIntergrate).LSTMOutputGateValueBuffer.begin(), (*ModuleToIntergrate).LSTMOutputGateValueBuffer.end());
	}

	OutputModule.LSTMCellStateValueBuffer.clear();
	if ((*InputModule).LSTMCellStateValueBuffer.size() != 0) {
		OutputModule.LSTMCellStateValueBuffer.insert(OutputModule.LSTMCellStateValueBuffer.begin(), (*InputModule).LSTMCellStateValueBuffer.begin(), (*InputModule).LSTMCellStateValueBuffer.end());
	}
	if ((*ModuleToIntergrate).LSTMCellStateValueBuffer.size() != 0) {
		OutputModule.LSTMCellStateValueBuffer.insert(OutputModule.LSTMCellStateValueBuffer.end(), (*ModuleToIntergrate).LSTMCellStateValueBuffer.begin(), (*ModuleToIntergrate).LSTMCellStateValueBuffer.end());
	}

	OutputModule.LSTMOutputValueBuffer.clear();
	if ((*InputModule).LSTMOutputValueBuffer.size() != 0) {
		OutputModule.LSTMOutputValueBuffer.insert(OutputModule.LSTMOutputValueBuffer.begin(), (*InputModule).LSTMOutputValueBuffer.begin(), (*InputModule).LSTMOutputValueBuffer.end());
	}
	if ((*ModuleToIntergrate).LSTMOutputValueBuffer.size() != 0) {
		OutputModule.LSTMOutputValueBuffer.insert(OutputModule.LSTMOutputValueBuffer.end(), (*ModuleToIntergrate).LSTMOutputValueBuffer.begin(), (*ModuleToIntergrate).LSTMOutputValueBuffer.end());
	}

	//NeuronInputNodeStartBuffer
	if (((*InputModule).NeuronNodeStartBuffer.size() == 0) && ((*ModuleToIntergrate).NeuronNodeStartBuffer.size() == 0)) {
		OutputModule.NeuronNodeStartBuffer.clear();
	}else {
		IntergrateUnsignedIntVectorNeuron(
			&(*InputModule).NeuronNodeStartBuffer,
			&(*ModuleToIntergrate).NeuronNodeStartBuffer,
			OutputModule.NeuronNodeStartBuffer
		);
	}

	//NeuronInputNodeEndBuffer
	if (((*InputModule).NeuronNodeEndBuffer.size() == 0) && ((*ModuleToIntergrate).NeuronNodeEndBuffer.size() == 0)) {
		OutputModule.NeuronNodeEndBuffer.clear();
	}else {
		IntergrateUnsignedIntVectorNeuron(
			&(*InputModule).NeuronNodeEndBuffer,
			&(*ModuleToIntergrate).NeuronNodeEndBuffer,
			OutputModule.NeuronNodeEndBuffer
		);
	}

	//IntergrationNeuronInputNodeIDsBuffer
	OutputModule.NeuronInputNodeIDsBuffer.resize((*InputModule).NeuronInputNodeIDsBuffer.size() + (*ModuleToIntergrate).NeuronInputNodeIDsBuffer.size());
	copy((*InputModule).NeuronInputNodeIDsBuffer.begin(), (*InputModule).NeuronInputNodeIDsBuffer.end(), OutputModule.NeuronInputNodeIDsBuffer.begin());
	for (unsigned int i = 0; i < (*ModuleToIntergrate).NeuronInputNodeIDsBuffer.size(); i++) {
		if ((NetworkModule == 1) && ((*ModuleToIntergrate).NeuronInputNodeIDsBuffer[i] <= NumofNNInputs)) {
			OutputModule.NeuronInputNodeIDsBuffer.push_back(0);
		}else {
			OutputModule.NeuronInputNodeIDsBuffer.push_back((*ModuleToIntergrate).NeuronInputNodeIDsBuffer[i] + BufferNodeOffset);
		}
	}

	//IntergrationNeuronOutputNodeIDsBuffer
	OutputModule.NeuronOutputNodeIDsBuffer.resize((*InputModule).NeuronOutputNodeIDsBuffer.size() + (*ModuleToIntergrate).NeuronOutputNodeIDsBuffer.size());
	copy((*InputModule).NeuronOutputNodeIDsBuffer.begin(), (*InputModule).NeuronOutputNodeIDsBuffer.end(), OutputModule.NeuronOutputNodeIDsBuffer.begin());
	for (unsigned int i = 0; i < (*ModuleToIntergrate).NeuronOutputNodeIDsBuffer.size(); i++) {
		if ((NetworkModule == 1) && ((*ModuleToIntergrate).NeuronOutputNodeIDsBuffer[i] <= NumofNNInputs)) {
			OutputModule.NeuronOutputNodeIDsBuffer.push_back(0);
		}else {
			OutputModule.NeuronOutputNodeIDsBuffer.push_back((*ModuleToIntergrate).NeuronOutputNodeIDsBuffer[i] + BufferNodeOffset);
		}
	}


	//NeuronOutputNodeBuffer
	if (((*InputModule).NeuronOutputNodeBuffer.size() == 0) && ((*ModuleToIntergrate).NeuronOutputNodeBuffer.size() == 0)) {
		OutputModule.NeuronNodeEndBuffer.clear();
	}else {
		IntergrateUnsignedFloatVectorNeuron(
			&(*InputModule).NeuronOutputNodeBuffer,
			&(*ModuleToIntergrate).NeuronOutputNodeBuffer,
			OutputModule.NeuronOutputNodeBuffer
		);
	}


	//NeuronWeightsBuffer
	if (((*InputModule).NeuronWeightsBuffer.size() == 0) && ((*ModuleToIntergrate).NeuronWeightsBuffer.size() == 0)) {
		OutputModule.NeuronWeightsBuffer.clear();
	}else {
		IntergrateUnsignedFloatVectorNeuron(
			&(*InputModule).NeuronWeightsBuffer,
			&(*ModuleToIntergrate).NeuronWeightsBuffer,
			OutputModule.NeuronWeightsBuffer
		);
	}

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

int SeperateNeuralNetworkModule(
	OLD_NeuralNetworkModule *InputModule,
	OLD_NeuralNetworkModule &BaseOutputModule,
	OLD_NeuralNetworkModule &SeperatedOutputModule,

	unsigned int NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int BufferNodeOffset,
	unsigned int LSTMOffset,
	unsigned int NeuronOffset,
	unsigned int NeuronWeightSize
) {
	vector<float> SeperateBufferNodeValues;
	vector<unsigned int> SeperateLSTMInputNodeIDBuffer;
	vector<unsigned int> SeperateLSTMOutputNodeIDBuffer;

	vector<unsigned int> SeperateNeuronInputNodeIDsBuffer;
	vector<unsigned int> SeperateNeuronOutputNodeIDsBuffer;
	NeuronOffset = (*InputModule).NeuronNodeStartBuffer.size();

	//BufferNodeValues
	if ((*InputModule).BufferNodeValues.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).BufferNodeValues,
			BaseOutputModule.BufferNodeValues,
			SeperatedOutputModule.BufferNodeValues,
			LSTMOffset
		);
	}

	//LSTMInputNodeIDBuffer
	if ((*InputModule).LSTMInputNodeIDBuffer.size() != 0) {
		while (BaseOutputModule.LSTMInputNodeIDBuffer.size() != 0) {
			BaseOutputModule.LSTMInputNodeIDBuffer.clear();
		}
		BaseOutputModule.LSTMInputNodeIDBuffer.insert(BaseOutputModule.LSTMInputNodeIDBuffer.begin(), (*InputModule).LSTMInputNodeIDBuffer.begin(), (*InputModule).LSTMInputNodeIDBuffer.begin() + LSTMOffset);
		for (unsigned int i = 0; i < (*InputModule).LSTMInputNodeIDBuffer.size() - LSTMOffset; i++) {
			if ((NetworkModule == 1) && ((*InputModule).LSTMInputNodeIDBuffer[i] <= NumofNNInputs)) {
				SeperateLSTMInputNodeIDBuffer.push_back((*InputModule).LSTMInputNodeIDBuffer[i]);
			}else {
				SeperateLSTMInputNodeIDBuffer.push_back((*InputModule).LSTMInputNodeIDBuffer[i] - BufferNodeOffset);
			}
		}
		while (SeperatedOutputModule.LSTMInputNodeIDBuffer.size() != 0) {
			SeperatedOutputModule.LSTMInputNodeIDBuffer.clear();
		}
		SeperatedOutputModule.LSTMInputNodeIDBuffer.insert(SeperatedOutputModule.LSTMInputNodeIDBuffer.begin(), SeperateLSTMInputNodeIDBuffer.begin(), SeperateLSTMInputNodeIDBuffer.end());
	}

	//LSTMOutputNodeIDBuffer
	if ((*InputModule).LSTMOutputNodeIDBuffer.size() != 0) {
		while (BaseOutputModule.LSTMOutputNodeIDBuffer.size() != 0) {
			BaseOutputModule.LSTMOutputNodeIDBuffer.clear();
		}
		BaseOutputModule.LSTMOutputNodeIDBuffer.insert(BaseOutputModule.LSTMOutputNodeIDBuffer.begin(), (*InputModule).LSTMOutputNodeIDBuffer.begin(), (*InputModule).LSTMOutputNodeIDBuffer.begin() + LSTMOffset);
		for (unsigned int i = 0; i < (*InputModule).LSTMOutputNodeIDBuffer.size() - LSTMOffset; i++) {
			if ((NetworkModule == 1) && ((*InputModule).LSTMOutputNodeIDBuffer[i] <= NumofNNInputs)) {
				SeperateLSTMOutputNodeIDBuffer.push_back((*InputModule).LSTMOutputNodeIDBuffer[i]);
			}else {
				SeperateLSTMOutputNodeIDBuffer.push_back((*InputModule).LSTMOutputNodeIDBuffer[i] - BufferNodeOffset);
			}
		}
		while (SeperatedOutputModule.LSTMOutputNodeIDBuffer.size() != 0) {
			SeperatedOutputModule.LSTMOutputNodeIDBuffer.clear();
		}
		SeperatedOutputModule.LSTMOutputNodeIDBuffer.insert(SeperatedOutputModule.LSTMOutputNodeIDBuffer.begin(), SeperateLSTMOutputNodeIDBuffer.begin(), SeperateLSTMOutputNodeIDBuffer.end());
	}

	//LSTMWeightsListBuffer
	if ((*InputModule).LSTMWeightsListBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).LSTMWeightsListBuffer,
			BaseOutputModule.LSTMWeightsListBuffer,
			SeperatedOutputModule.LSTMWeightsListBuffer,
			(LSTMOffset * 8)
		);
	}

	//LSTMPreviousCellStateValueBuffer
	if ((*InputModule).LSTMPreviousCellStateValueBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).LSTMPreviousCellStateValueBuffer,
			BaseOutputModule.LSTMPreviousCellStateValueBuffer,
			SeperatedOutputModule.LSTMPreviousCellStateValueBuffer,
			LSTMOffset
		);
	}

	//LSTMPreviousOutputValueBuffer
	if ((*InputModule).LSTMPreviousOutputValueBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).LSTMPreviousOutputValueBuffer,
			BaseOutputModule.LSTMPreviousOutputValueBuffer,
			SeperatedOutputModule.LSTMPreviousOutputValueBuffer,
			LSTMOffset
		);
	}

	//LSTMForgetGateValueBuffer
	if ((*InputModule).LSTMForgetGateValueBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).LSTMForgetGateValueBuffer,
			BaseOutputModule.LSTMForgetGateValueBuffer,
			SeperatedOutputModule.LSTMForgetGateValueBuffer,
			LSTMOffset
		);
	}

	//LSTMInputGateValueBuffer
	if ((*InputModule).LSTMInputGateValueBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).LSTMInputGateValueBuffer,
			BaseOutputModule.LSTMInputGateValueBuffer,
			SeperatedOutputModule.LSTMInputGateValueBuffer,
			LSTMOffset
		);
	}

	//LSTMCandidateValueBuffer
	if ((*InputModule).LSTMCandidateValueBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).LSTMCandidateValueBuffer,
			BaseOutputModule.LSTMCandidateValueBuffer,
			SeperatedOutputModule.LSTMCandidateValueBuffer,
			LSTMOffset
		);
	}

	//LSTMOutputGateValueBuffer
	if ((*InputModule).LSTMOutputGateValueBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).LSTMOutputGateValueBuffer,
			BaseOutputModule.LSTMOutputGateValueBuffer,
			SeperatedOutputModule.LSTMOutputGateValueBuffer,
			LSTMOffset
		);
	}

	//LSTMCellStateValueBuffer
	if ((*InputModule).LSTMCellStateValueBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).LSTMCellStateValueBuffer,
			BaseOutputModule.LSTMCellStateValueBuffer,
			SeperatedOutputModule.LSTMCellStateValueBuffer,
			LSTMOffset
		);
	}

	//LSTMOutputValueBuffer
	if ((*InputModule).LSTMOutputValueBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).LSTMOutputValueBuffer,
			BaseOutputModule.LSTMOutputValueBuffer,
			SeperatedOutputModule.LSTMOutputValueBuffer,
			LSTMOffset
		);
	}

	//NeuronInputNodeStartBuffer
	if ((*InputModule).NeuronNodeStartBuffer.size() != 0) {
		SeperateLSTMUInt(
			&(*InputModule).NeuronNodeStartBuffer,
			BaseOutputModule.NeuronNodeStartBuffer,
			SeperatedOutputModule.NeuronNodeStartBuffer,
			NeuronOffset
		);
	}

	//NeuronInputNodeEndBuffer
	if ((*InputModule).NeuronNodeEndBuffer.size() != 0) {
		SeperateLSTMUInt(
			&(*InputModule).NeuronNodeEndBuffer,
			BaseOutputModule.NeuronNodeEndBuffer,
			SeperatedOutputModule.NeuronNodeEndBuffer,
			NeuronOffset
		);
	}

	//NeuronInputNodeIDsBuffer
	if ((*InputModule).NeuronInputNodeIDsBuffer.size() != 0) {
		while (BaseOutputModule.NeuronInputNodeIDsBuffer.size() != 0) {
			BaseOutputModule.NeuronInputNodeIDsBuffer.clear();
		}
		BaseOutputModule.NeuronInputNodeIDsBuffer.insert(BaseOutputModule.NeuronInputNodeIDsBuffer.begin(), (*InputModule).NeuronInputNodeIDsBuffer.begin(), (*InputModule).NeuronInputNodeIDsBuffer.begin() + NeuronOffset);
		for (unsigned int i = 0; i < (*InputModule).NeuronInputNodeIDsBuffer.size() - NeuronOffset; i++) {
			if ((NetworkModule == 1) && ((*InputModule).NeuronInputNodeIDsBuffer[i] <= NumofNNInputs)) {
				SeperateNeuronInputNodeIDsBuffer.push_back((*InputModule).NeuronInputNodeIDsBuffer[i]);
			}else {
				SeperateNeuronInputNodeIDsBuffer.push_back((*InputModule).NeuronInputNodeIDsBuffer[i] - BufferNodeOffset);
			}
		}
		while (SeperatedOutputModule.NeuronInputNodeIDsBuffer.size() != 0) {
			SeperatedOutputModule.NeuronInputNodeIDsBuffer.clear();
		}
		SeperatedOutputModule.NeuronInputNodeIDsBuffer.insert(SeperatedOutputModule.NeuronInputNodeIDsBuffer.begin(), SeperateNeuronInputNodeIDsBuffer.begin(), SeperateNeuronInputNodeIDsBuffer.end());
	}

	//NeuronOutputNodeIDsBuffer
	if ((*InputModule).NeuronOutputNodeIDsBuffer.size() != 0) {
		while (BaseOutputModule.NeuronOutputNodeIDsBuffer.size() != 0) {
			BaseOutputModule.NeuronOutputNodeIDsBuffer.clear();
		}
		BaseOutputModule.NeuronOutputNodeIDsBuffer.insert(BaseOutputModule.NeuronOutputNodeIDsBuffer.begin(), (*InputModule).NeuronOutputNodeIDsBuffer.begin(), (*InputModule).NeuronOutputNodeIDsBuffer.begin() + NeuronOffset);
		for (unsigned int i = 0; i < (*InputModule).NeuronOutputNodeIDsBuffer.size() - NeuronOffset; i++) {
			if ((NetworkModule == 1) && ((*InputModule).NeuronOutputNodeIDsBuffer[i] <= NumofNNInputs)) {
				SeperateNeuronOutputNodeIDsBuffer.push_back((*InputModule).NeuronOutputNodeIDsBuffer[i]);
			}else {
				SeperateNeuronOutputNodeIDsBuffer.push_back((*InputModule).NeuronOutputNodeIDsBuffer[i] - BufferNodeOffset);
			}
		}
		while (SeperatedOutputModule.NeuronOutputNodeIDsBuffer.size() != 0) {
			SeperatedOutputModule.NeuronOutputNodeIDsBuffer.clear();
		}
		SeperatedOutputModule.NeuronOutputNodeIDsBuffer.insert(SeperatedOutputModule.NeuronOutputNodeIDsBuffer.begin(), SeperateNeuronOutputNodeIDsBuffer.begin(), SeperateNeuronOutputNodeIDsBuffer.end());
	}

	//NeuronOutputNodeBuffer
	if ((*InputModule).NeuronOutputNodeBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).NeuronOutputNodeBuffer,
			BaseOutputModule.NeuronOutputNodeBuffer,
			SeperatedOutputModule.NeuronOutputNodeBuffer,
			NeuronOffset
		);
	}

	//NeuronWeightsBuffer
	if ((*InputModule).NeuronWeightsBuffer.size() != 0) {
		SeperateLSTMFloat(
			&(*InputModule).NeuronWeightsBuffer,
			BaseOutputModule.NeuronWeightsBuffer,
			SeperatedOutputModule.NeuronWeightsBuffer,
			NeuronWeightSize
		);
	}

	return 1;
}


