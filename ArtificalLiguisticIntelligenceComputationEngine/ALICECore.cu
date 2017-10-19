#include "ALICECore.cuh"

__device__ float Sigmoid(float a) {
	return 1.0 / (1.0 + exp(-a));
}


__device__ float AIScoringFunction(
	float *NeuronOutputNodeBuffer,
	float *ScoringVariables
	) {
	/*
	ScoringVariables
	[0]TrainingMode
	[1]NumOfNodesToScore
	[2 + (n * 2)]NodeIDsToScore
	[3 + (n * 2)]ScoringVariables
	*/
	float NetworkScore = 0;
	unsigned int TrainingMode = (unsigned int)ScoringVariables[0];
	unsigned int NumOfNodesToScore = (unsigned int)ScoringVariables[1];
	float ScoreBuffer = 0;
	float ValueOfNodeBuffer = 0;
	float ValueOfScoringVariable = 0;

	//if (UniqueThreadID == NumOfNeurons) {	//Score the system
	//if (UniqueThreadID == 0) {	//Score the system
		//0=Emotinal training, 1=Offline trainer training , 2=Offline training, 3=Online training
		//Score the output
	switch (TrainingMode) {
	case 0://0=Emotinal training
		for (unsigned int n = 0; n < NumOfNodesToScore; n++) {
			ValueOfNodeBuffer = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (n * 2)]];
			ValueOfScoringVariable = ScoringVariables[3 + (n * 2)];
			ScoreBuffer = (ValueOfNodeBuffer - ValueOfScoringVariable);
			if (ScoreBuffer > 0) {
				ScoreBuffer = -ScoreBuffer;
			}
			NetworkScore += ScoreBuffer + 1.0;
		}
		break;
	case 1://1=Offline trainer training
		for (unsigned int n = 0; n < 8; n++) {
			ScoreBuffer = (ValueOfNodeBuffer - ValueOfScoringVariable) / 2.0;
			if (ScoreBuffer > 0) {
				ScoreBuffer = -ScoreBuffer;
			}
			NetworkScore += ScoreBuffer + 1.0;
		}
		break;
	case 2://2=Offline training
		for (unsigned int n = 0; n < 8; n++) {
			ScoreBuffer = (ValueOfNodeBuffer - ValueOfScoringVariable) / 2.0;
			if (ScoreBuffer > 0) {
				ScoreBuffer = -ScoreBuffer;
			}
			NetworkScore += ScoreBuffer + 1.0;
		}
		break;
	case 3://3 = Online training
		float Ecstasy = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (0 * 2)]];
		float Admiration = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (1 * 2)]];
		float Terror = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (2 * 2)]];
		float Amazement = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (3 * 2)]];
		float Grief = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (4 * 2)]];
		float Loathing = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (5 * 2)]];
		float Rage = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (6 * 2)]];
		float Vigilance = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (7 * 2)]];
		float ID = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (8 * 2)]];
		float Busy = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (9 * 2)]];
		float Present = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (10 * 2)]];
		float AdminPresent = ((0.375 * ID) + 0.625) * ((Present + 1.0) / 2.0);
		NetworkScore += Ecstasy * AdminPresent * 3.0;
		NetworkScore += Admiration * AdminPresent * 4.0;
		NetworkScore += Terror * AdminPresent * -4.0;
		NetworkScore += Amazement * AdminPresent * 2.0;
		NetworkScore += Grief * AdminPresent * -1.0;
		NetworkScore += Loathing * AdminPresent;
		NetworkScore += Rage * AdminPresent * -3.0;
		NetworkScore += Vigilance * AdminPresent * 1.0;
		break;
	}

		//NetworkScore *= 0.0001;
	//}
	return NetworkScore;
}

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
	3 = Multiply ((input[] * weight[]) + bias[])
	4 = Min ((input[] * weight[]) + bias[])
	5 = Max ((input[] * weight[]) + bias[])
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
){
	
	unsigned int UniqueThreadID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ThreadID = threadIdx.x;
	unsigned int BlockID = blockIdx.x;
	unsigned int Blockdim = blockDim.x;

	//Prepare variables for each thread
	unsigned int NumOfCycles = NeuralNetworkSetup[0];
	unsigned int NumOfInputs = NeuralNetworkSetup[1];
	unsigned int NumOfOutputs = NeuralNetworkSetup[2];
	unsigned int LocalVarNumOfNeurons = NumOfNeurons;

	float NodeBufferValue = 0.0;

	NetworkScore[0] = 0;

	//Map inputs to input nodes
	unsigned int InputNodeID = 0;
	unsigned int InputIncrementElementEveryXNumOfCylce = 0;
	unsigned int InputDataElementSize = 0;
	unsigned int InputDataStartArrayID = 0;
	unsigned int UniqueThreadIDX3 = UniqueThreadID * 3;
	
	//if (UniqueThreadID < 2210) {
	if (UniqueThreadID < NumOfInputs) {
		InputNodeID = NeuralNetworkSetup[4 + UniqueThreadIDX3];
		InputIncrementElementEveryXNumOfCylce = NeuralNetworkSetup[5 + UniqueThreadIDX3];
		InputDataElementSize = NeuralNetworkSetup[3 + UniqueThreadIDX3];
		//InputDataStartArrayID = NeuralNetworkSetup[3];
		//for (unsigned int i = 1; i < UniqueThreadID; i++) {
		for (unsigned int i = 0; i < UniqueThreadID; i++) {
			InputDataStartArrayID += NeuralNetworkSetup[3 + (i * 3)];
		}

	}
	//OutputData[0] = 0;
	//Map outputs to output nodes
	unsigned int OutputNodeID = 0;
	unsigned int OutputIncrementElementEveryXNumOfCylce = 0;
	unsigned int OutputDataElementSize = 0;
	unsigned int OutputDataStartArrayID = 0;
	unsigned int NumOfInputsX3 = NumOfInputs * 3;
	if (UniqueThreadID < NumOfOutputs) {
		OutputNodeID = NeuralNetworkSetup[4 + NumOfInputsX3 + UniqueThreadIDX3];
		OutputIncrementElementEveryXNumOfCylce = NeuralNetworkSetup[5 + NumOfInputsX3 + UniqueThreadIDX3];
		OutputDataElementSize = NeuralNetworkSetup[3 + NumOfInputsX3 + UniqueThreadIDX3];
		OutputDataStartArrayID = NeuralNetworkSetup[3 + NumOfInputsX3];
		for (unsigned int i = 1; i < UniqueThreadID; i++) {
			OutputDataStartArrayID += NeuralNetworkSetup[3 + NumOfInputsX3 + (i * 3)];
		}
	}
	__syncthreads();
	//for (unsigned int i = 0; i <100; i++) {//FRAMES_PER_BUFFER


	//Prepare variables for each thread
	unsigned int UniqueTheadNeuronNodeStartBuffer = 0;
	unsigned int UniqueTheadNeuronNodeLength = 0;
	unsigned int UniqueTheadTypeOfNode = 0;
	unsigned int UniqueTheadNeuronOutputNodeIDsBuffer = 0;
	if (UniqueThreadID < LocalVarNumOfNeurons) {
		UniqueTheadNeuronNodeStartBuffer = NeuronNodeStartBuffer[UniqueThreadID];
		UniqueTheadNeuronNodeLength = NeuronNodeLength[UniqueThreadID];
		UniqueTheadTypeOfNode = TypeOfNode[UniqueThreadID];
		UniqueTheadNeuronOutputNodeIDsBuffer = NeuronOutputNodeIDsBuffer[UniqueThreadID];
	}

	//Start running through each cycle
	unsigned int InputCycleCounter;
	unsigned int OutputCycleCounter;
	for (unsigned int i = 0; i < NumOfCycles; i++) {//FRAMES_PER_BUFFER
		InputCycleCounter = i / InputIncrementElementEveryXNumOfCylce;
		OutputCycleCounter = i / OutputIncrementElementEveryXNumOfCylce;
		////__syncthreads();
		//Copy input data to inputNodes
		if (UniqueThreadID < NumOfInputs) {
			if (InputIncrementElementEveryXNumOfCylce == 0) {
				BufferNodeValues[InputNodeID] = InputData[InputDataStartArrayID];
			}else if (InputDataElementSize > InputCycleCounter) {
				BufferNodeValues[InputNodeID] = InputData[InputDataStartArrayID + InputCycleCounter];
			}else {
				//Do not change the value
				//BufferNodeValues[InputNodeID] = InputDataElementSize;//2205
				//BufferNodeValues[InputNodeID] = InputCycleCounter;//2204
				BufferNodeValues[InputNodeID] = InputData[InputDataStartArrayID + (InputDataElementSize - 1)];
			}
		}
		__syncthreads();

		if (UniqueThreadID < LocalVarNumOfNeurons) {
			/*
			if (TypeOfNode[UniqueThreadID] == 3) {
				NodeBufferValue = 1.0;	//Used for multiplication, don't want to start with zero... 
			}else if (TypeOfNode[UniqueThreadID] == 4) {
				NodeBufferValue = NPP_MAXABS_32F;
			}else if (TypeOfNode[UniqueThreadID] == 5) {
				NodeBufferValue = NPP_MINABS_32F;
			}else {
				NodeBufferValue = 0.0;
			}*/
			switch (UniqueTheadTypeOfNode) {
			case 3:
				NodeBufferValue = 1.0;	//Used for multiplication, don't want to start with zero... 
				break;
			case 4:
				NodeBufferValue = NPP_MAXABS_32F;
				break;
			case 5:
				NodeBufferValue = NPP_MINABS_32F;
				break;
			default:
				NodeBufferValue = 0.0;
				break;
			}


			unsigned int InputNodeID = 0;
			for (unsigned int n = 0; n < UniqueTheadNeuronNodeLength; n++) {
				InputNodeID = (UniqueTheadNeuronNodeStartBuffer + n);

				/*
				if (TypeOfNode[UniqueThreadID] == 0) {// +
					NodeBufferValue += (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					//NodeBufferValue += BufferNodeValues[InputNodeID];
				}else if (TypeOfNode[UniqueThreadID] == 1) {//Sigmoid
					NodeBufferValue += (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					NodeBufferValue = Sigmoid(NodeBufferValue);
				}else if (TypeOfNode[UniqueThreadID] == 2) {//Tanh
					NodeBufferValue += (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					NodeBufferValue = tanh(NodeBufferValue);
				}else if (TypeOfNode[UniqueThreadID] == 3) {// *
					NodeBufferValue *= (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					//NodeBufferValue *= BufferNodeValues[InputNodeID];
				}else if (TypeOfNode[UniqueThreadID] == 4) { // Min
					if ((NodeBufferValue > BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID]) {
						NodeBufferValue = (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					}
				}else if (TypeOfNode[UniqueThreadID] == 5) { // Max
					if ((NodeBufferValue < BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID]) {
						NodeBufferValue = (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					}
				}*/

				float NodeInputBufferValue = (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];

				switch (UniqueTheadTypeOfNode) {
				case 0:// +
					NodeBufferValue += NodeInputBufferValue;
					break;
				case 1:// Sigmoid
					NodeBufferValue += NodeInputBufferValue;
				//	if (NodeBufferValue == 1.0) {
				//		NodeBufferValue = (float)0.731058579;
				//	}else if (NodeBufferValue == -1.0) {
				//		NodeBufferValue = (float)0.268941421;
				//	}else {
						NodeBufferValue = Sigmoid(NodeBufferValue);
				//	}
					break;
				case 2:// Tanh
					NodeBufferValue += NodeInputBufferValue;
				//	if (NodeBufferValue == 1.0) {
				//		NodeBufferValue = (float)0.76159415595;
				//	}else if (NodeBufferValue == -1.0) {
				//		NodeBufferValue = (float)-0.76159415595;
				//	}else {
						NodeBufferValue = tanh(NodeBufferValue);
				//	}



					break;
				case 3:// *
					NodeBufferValue *= NodeInputBufferValue;
					break;
				case 4:// Min
					if (NodeBufferValue > NodeInputBufferValue) {
						NodeBufferValue = NodeInputBufferValue;
					}
					break;
				case 5:// Max
					if (NodeBufferValue < NodeInputBufferValue) {
						NodeBufferValue = NodeInputBufferValue;
					}
					break;
				}

				if (NodeBufferValue > 1.0) {
					NodeBufferValue = 1.0;
				}else if (NodeBufferValue < -1.0) {
					NodeBufferValue = -1.0;
				}
				
			}
			NeuronOutputNodeBuffer[UniqueThreadID] = NodeBufferValue;

		}
		
		__syncthreads();
		if (UniqueThreadID < LocalVarNumOfNeurons) {			//Transfer the value buffers
			//BufferNodeValues[NeuronOutputNodeIDsBuffer[UniqueThreadID]] = NeuronOutputNodeBuffer[UniqueThreadID];
			BufferNodeValues[UniqueTheadNeuronOutputNodeIDsBuffer] = NodeBufferValue;
		}else if (UniqueThreadID == LocalVarNumOfNeurons) {
			NetworkScore[0] = AIScoringFunction(
				NeuronOutputNodeBuffer,
				ScoringVariables
			);
		}
		__syncthreads();
		
		//Copy outputNodes data to outputs
		if (UniqueThreadID < NumOfOutputs) {
			if (OutputIncrementElementEveryXNumOfCylce == 0) {
				OutputData[OutputDataStartArrayID] = BufferNodeValues[OutputNodeID];
			}else if (OutputDataElementSize > OutputCycleCounter) {
				OutputData[OutputDataStartArrayID + OutputCycleCounter] = BufferNodeValues[OutputNodeID];
			}else {
				OutputData[OutputDataStartArrayID + (OutputDataElementSize - 1)] = BufferNodeValues[OutputNodeID];
			}
		}
		//__syncthreads();
		
	}
	//
	
	//NetworkScore[0] *= 0.0001; //Divide score by 10000 to prevent score becoming too large in Core AI thread
}

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
){
	unsigned int dev_NeuralNetworkSetupSize = 0;
	unsigned int *dev_NeuralNetworkSetup = 0;
	unsigned int dev_InputDataSize = 0;
	float *dev_InputData = 0;
	unsigned int dev_OutputDataSize = 0;
	float *dev_OutputData = 0;
	unsigned int dev_NumOfBufferNodes = 0;
	float *dev_BufferNodeValues = 0;
	unsigned int dev_NumOfNeurons = 0;
	unsigned int *dev_TypeOfNode = 0;
	unsigned int dev_NumOfNeuronInputs = 0;
	unsigned int *dev_NeuronNodeStartBuffer = 0;
	unsigned int *dev_NeuronNodeLength = 0;
	unsigned int *dev_NeuronInputNodeIDsBuffer = 0;
	unsigned int *dev_NeuronOutputNodeIDsBuffer = 0;
	float *dev_NeuronOutputNodeBuffer = 0;
	unsigned int dev_NumOfNeuronWeights = 0;
	float *dev_NeuronWeightsBuffer = 0;
	float *dev_NeuronBiasBuffer = 0;
	unsigned int dev_NumOfScoringVariables = 0;
	float *dev_ScoringVariables = 0;
	float *dev_NetworkScore = 0;
	cudaError_t cudaStatus;

	dev_NumOfBufferNodes = NumOfBufferNodes;
	dev_NumOfNeurons = NumOfNeurons;
	dev_NumOfNeuronInputs = NumOfNeuronInputs;
	dev_NumOfScoringVariables = NumOfScoringVariables;
	dev_NeuralNetworkSetupSize = NeuralNetworkSetupSize;
	dev_InputDataSize = InputDataSize;
	dev_OutputDataSize = OutputDataSize;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_NeuralNetworkSetup, dev_NeuralNetworkSetupSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NeuralNetworkSetup");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_InputData, dev_InputDataSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_InputData");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_OutputData, dev_OutputDataSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_OutputData");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_BufferNodeValues, dev_NumOfBufferNodes * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_BufferNodeValues");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_TypeOfNode, dev_NumOfNeurons * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_TypeOfNode");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NeuronNodeStartBuffer, dev_NumOfNeurons * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NeuronInputNodeStartBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NeuronNodeLength, dev_NumOfNeurons * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NeuronInputNodeEndBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NeuronInputNodeIDsBuffer, dev_NumOfNeuronInputs * sizeof(unsigned int));//FIX IT
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NeuronInputNodeIDsBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NeuronOutputNodeIDsBuffer, dev_NumOfNeurons * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NeuronOutputNodeIDsBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NeuronOutputNodeBuffer, dev_NumOfNeurons * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NeuronOutputNodeBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NeuronWeightsBuffer, NumOfNeuronWeights * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NeuronWeightsBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NeuronBiasBuffer, NumOfNeuronWeights * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NeuronBiasBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_ScoringVariables, dev_NumOfScoringVariables * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_ScoringVariables");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NetworkScore, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NetworkScore");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_NeuralNetworkSetup, NeuralNetworkSetup, dev_NeuralNetworkSetupSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_NeuralNetworkSetup");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_InputData, InputData, dev_InputDataSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_InputData");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_BufferNodeValues, BufferNodeValues, dev_NumOfBufferNodes * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_BufferNodeValues");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_TypeOfNode, TypeOfNode, dev_NumOfNeurons * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_TypeOfNode");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_NeuronNodeStartBuffer, NeuronNodeStartBuffer, dev_NumOfNeurons * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_NeuronInputNodeStartBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_NeuronNodeLength, NeuronNodeLength, dev_NumOfNeurons * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_NeuronInputNodeEndBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_NeuronInputNodeIDsBuffer, NeuronInputNodeIDsBuffer, dev_NumOfNeuronInputs * sizeof(unsigned int), cudaMemcpyHostToDevice);//FIX IT
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_NeuronInputNodeIDsBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_NeuronOutputNodeIDsBuffer, NeuronOutputNodeIDsBuffer, dev_NumOfNeurons * sizeof(unsigned int), cudaMemcpyHostToDevice);//FIX IT
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_NeuronOutputNodeIDsBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_NeuronOutputNodeBuffer, NeuronOutputNodeBuffer, dev_NumOfNeurons * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_NeuronOutputNodeBuffer");
		goto Error;
	}

	
	
	cudaStatus = cudaMemcpy(dev_NeuronWeightsBuffer, NeuronWeightsBuffer, NumOfNeuronWeights * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_NeuronWeightsBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_NeuronBiasBuffer, NeuronBiasBuffer, NumOfNeuronWeights * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_NeuronBiasBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_ScoringVariables, ScoringVariables, dev_NumOfScoringVariables * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_ScoringVariables");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	CoreAIKernel << <1, 256 >> > (
		dev_NeuralNetworkSetup,
		dev_InputData,
		dev_OutputData,
		dev_BufferNodeValues,
		dev_NumOfNeurons,
		/*
		0 = Sum ((input[] * weight[]) + bias[])
		1 = Sigmoid (Sum ((input[] * weight[]) + bias[]))
		2 = Tanh (Sum ((input[] * weight[]) + bias[]))
		3 = Sum
		4 = Multiply
		5 = Min
		6 = Max
		*/
		dev_TypeOfNode,
		dev_NeuronNodeStartBuffer,
		dev_NeuronNodeLength,
		dev_NeuronInputNodeIDsBuffer,
		dev_NeuronOutputNodeIDsBuffer,
		dev_NeuronOutputNodeBuffer,
		dev_NeuronWeightsBuffer,
		dev_NeuronBiasBuffer,
		dev_ScoringVariables,
		dev_NetworkScore
	);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ALICECoreKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching ALICECoreKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(OutputData, dev_OutputData, dev_OutputDataSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(BufferNodeValues, dev_BufferNodeValues, dev_NumOfBufferNodes * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(NetworkScore, dev_NetworkScore, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_NeuralNetworkSetup);
	cudaFree(dev_InputData);
	cudaFree(dev_OutputData);
	cudaFree(dev_BufferNodeValues);
	cudaFree(dev_TypeOfNode);
	cudaFree(dev_NeuronNodeStartBuffer);
	cudaFree(dev_NeuronNodeLength);
	cudaFree(dev_NeuronInputNodeIDsBuffer);
	cudaFree(dev_NeuronOutputNodeIDsBuffer);
	cudaFree(dev_NeuronOutputNodeBuffer);
	cudaFree(dev_NeuronWeightsBuffer);
	cudaFree(dev_NeuronBiasBuffer);
	cudaFree(dev_ScoringVariables);
	cudaFree(dev_NetworkScore);
	return cudaStatus;
}

/*
int RunAICore(
	NeuralNetworkModule& NeuralNetworkIndividual,
	vector<float>& InputData,
	vector<float>& OutputData,
	vector<float>& ScoringVariables,
	float& NetworkScore
) {
	cudaError_t cudaStatus = RunAICoreOnGPU(
		NeuralNetworkIndividual.NeuralNetworkSetup.size(),
		&NeuralNetworkIndividual.NeuralNetworkSetup[0],
		InputData.size(),
		&InputData[0],
		OutputData.size(),
		&OutputData[0],
		NeuralNetworkIndividual.BufferNodeValues.size(),
		&NeuralNetworkIndividual.BufferNodeValues[0],
		NeuralNetworkIndividual.NeuronOutputNodeBuffer.size(),
		NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.size(),
		&NeuralNetworkIndividual.TypeOfNode[0],
		&NeuralNetworkIndividual.NeuronNodeStartBuffer[0],
		&NeuralNetworkIndividual.NeuronNodeLength[0],
		&NeuralNetworkIndividual.NeuronInputNodeIDsBuffer[0],
		&NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer[0],
		&NeuralNetworkIndividual.NeuronOutputNodeBuffer[0],
		NeuralNetworkIndividual.NeuronWeightsBuffer.size(),
		&NeuralNetworkIndividual.NeuronWeightsBuffer[0],
		&NeuralNetworkIndividual.NeuronBiasBuffer[0],
		ScoringVariables.size(),
		&ScoringVariables[0],
		&NetworkScore
	);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "RunALICECore failed!");
		return 1;
	}

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
}
*/



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
	float &ForgetGateValue)					//ft
{
	ForgetGateValue = Sigmoid((InputValue * ForgetGateInputWeight) + (PreviousOutputValue * ForgetGatePreviousInputWeight) + FogetGateBias);
}

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
	float &InputGateValue)					//it
{
	InputGateValue = Sigmoid((InputValue * InputGateInputWeight) + (PreviousOutputValue * InputGatePreviousInputWeight) + InputGateBias);
}

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
	float &CandidateValue)			//~ct
{
	CandidateValue = tanh((InputValue * CandidateGateInputWeight) + (PreviousOutputValue * CandidateGatePreviousInputWeight) + CandidateGateBias);
}

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
)
{
	OutputGateValue = Sigmoid((InputValue * OutputGateInputWeight) + (PreviousOutputValue * OutputGatePreviousInputWeight) + OutputGateBias);
}

__device__ void RunLSTMUnitStageTwoCellState(
	float PreviousCellStateValue,	//ct-1
	float ForgetGateValue,			//ft
	float InputGateValue,			//it
	float CandidateValue,			//~ct
	float &CellStateValue			//ct
)
{
	CellStateValue = (ForgetGateValue * PreviousCellStateValue) + (InputGateValue * CandidateValue);
}

__device__ void RunLSTMUnitStageThree(
	float OutputGateValue,			//ot
	float CellStateValue,			//ct
	float &OutputValue)				//ht
{
	OutputValue = OutputGateValue * tanh(CellStateValue);
}

