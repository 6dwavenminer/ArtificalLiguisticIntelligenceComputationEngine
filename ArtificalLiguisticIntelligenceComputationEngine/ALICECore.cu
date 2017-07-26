#include "ALICECore.cuh"

__device__ float Sigmoid(float a) {
	return 1.0 / (1.0 + exp(-a));
}


__device__ void AIScoringFunction(
	unsigned int UnqiueTheadID,
	unsigned int NumOfNeurons,
	float *NeuronOutputNodeBuffer,
	float *ScoringVariables,
	float &NetworkScore
	) {
	/*
	ScoringVariables
	[0]TrainingMode
	[1]NumOfNodesToScore
	[2 + (n * 2)]NodeIDsToScore
	[3 + (n * 2)]ScoringVariables
	*/

	unsigned int TrainingMode = (unsigned int)ScoringVariables[0];
	unsigned int NumOfNodesToScore = (unsigned int)ScoringVariables[1];
	float ScoreBuffer = 0;
	float ValueOfNodeBuffer;
	float ValueOfScoringVariable;

	if (UnqiueTheadID == NumOfNeurons) {	//Score the system
		//0=Emotinal training, 1=Offline trainer training , 2=Offline training, 3=Online training
		//Score the output
		if (TrainingMode == 0) {		//0=Emotinal training
			for (unsigned int n = 0; n < NumOfNodesToScore; n++) {
				ValueOfNodeBuffer = NeuronOutputNodeBuffer[(unsigned int)ScoringVariables[2 + (n * 2)]];
				ValueOfScoringVariable = ScoringVariables[3 + (n * 2)];
				ScoreBuffer = (ValueOfNodeBuffer - ValueOfScoringVariable);
				if (ScoreBuffer > 0) {
					ScoreBuffer = -ScoreBuffer;
				}
				NetworkScore += ScoreBuffer + 1.0;
			}
		}else if (TrainingMode == 1) {	//1=Offline trainer training
			for (unsigned int n = 0; n < 8; n++) {
				ScoreBuffer = (ValueOfNodeBuffer - ValueOfScoringVariable) / 2.0;
				if (ScoreBuffer > 0) {
					ScoreBuffer = -ScoreBuffer;
				}
				NetworkScore += ScoreBuffer + 1.0;
			}

		}else if (TrainingMode == 2) {	//2=Offline training
			for (unsigned int n = 0; n < 8; n++) {
				ScoreBuffer = (ValueOfNodeBuffer - ValueOfScoringVariable) / 2.0;
				if (ScoreBuffer > 0) {
					ScoreBuffer = -ScoreBuffer;
				}
				NetworkScore += ScoreBuffer + 1.0;
			}

		}else if (TrainingMode == 3) {	//3 = Online training
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

		}
	}
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
	
	unsigned int UnqiueTheadID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ThreadID = threadIdx.x;
	unsigned int BlockID = blockIdx.x;
	unsigned int Blockdim = blockDim.x;

	float NodeBufferValue = 0.0;

	NetworkScore[0] = 0;

	//Map inputs to input nodes
	unsigned int InputNodeID = 0;
	unsigned int InputIncrementElementEveryXNumOfCylce = 0;
	unsigned int InputDataElementSize = 0;
	unsigned int InputDataStartArrayID = 0;

	//if (UnqiueTheadID < 2210) {
	if (UnqiueTheadID < NeuralNetworkSetup[1]) {
		InputNodeID = NeuralNetworkSetup[4 + (UnqiueTheadID * 3)];
		InputIncrementElementEveryXNumOfCylce = NeuralNetworkSetup[5 + (UnqiueTheadID * 3)];
		InputDataElementSize = NeuralNetworkSetup[3 + (UnqiueTheadID * 3)];
		//InputDataStartArrayID = NeuralNetworkSetup[3];
		//for (unsigned int i = 1; i < UnqiueTheadID; i++) {
		for (unsigned int i = 0; i < UnqiueTheadID; i++) {
			InputDataStartArrayID += NeuralNetworkSetup[3 + (i * 3)];
		}

	}
	//OutputData[0] = 0;
	//Map outputs to output nodes
	unsigned int OutputNodeID = 0;
	unsigned int OutputIncrementElementEveryXNumOfCylce = 0;
	unsigned int OutputDataElementSize = 0;
	unsigned int OutputDataStartArrayID = 0;
	if (UnqiueTheadID < NeuralNetworkSetup[2]) {
		OutputNodeID = NeuralNetworkSetup[4 + (NeuralNetworkSetup[1] * 3) + (UnqiueTheadID * 3)];
		OutputIncrementElementEveryXNumOfCylce = NeuralNetworkSetup[5 + (NeuralNetworkSetup[1] * 3) + (UnqiueTheadID * 3)];
		OutputDataElementSize = NeuralNetworkSetup[3 + (NeuralNetworkSetup[1] * 3) + (UnqiueTheadID * 3)];
		OutputDataStartArrayID = NeuralNetworkSetup[3 + (NeuralNetworkSetup[1] * 3)];
		for (unsigned int i = 1; i < UnqiueTheadID; i++) {
			OutputDataStartArrayID += NeuralNetworkSetup[3 + (NeuralNetworkSetup[1] * 3) + (i * 3)];
		}
	}
	
	for (unsigned int i = 0; i < NeuralNetworkSetup[0]; i++) {//FRAMES_PER_BUFFER
		__syncthreads();
		//Copy input data to inputNodes
		if (UnqiueTheadID < NeuralNetworkSetup[1]) {
			if (InputIncrementElementEveryXNumOfCylce == 0) {
				BufferNodeValues[InputNodeID] = InputData[InputDataStartArrayID];
			}else if (InputDataElementSize < i / InputIncrementElementEveryXNumOfCylce) {
				BufferNodeValues[InputNodeID] = InputData[InputDataStartArrayID + (i / InputIncrementElementEveryXNumOfCylce)];
			}else {
				BufferNodeValues[InputNodeID] = InputData[InputDataStartArrayID + InputDataElementSize];
			}
		}
		__syncthreads();

		if (UnqiueTheadID < NumOfNeurons) {
			if (TypeOfNode[UnqiueTheadID] == 4) {
				NodeBufferValue = 1.0;	//Used for multiplication, don't want to start with zero... 
			}else if (TypeOfNode[UnqiueTheadID] == 5) {
				NodeBufferValue = NPP_MAXABS_32F;
			}else if (TypeOfNode[UnqiueTheadID] == 6) {
				NodeBufferValue = NPP_MINABS_32F;
			}else {
				NodeBufferValue = 0.0;
			}
			unsigned int InputNodeID = 0;
			for (unsigned int n = 0; n < NeuronNodeLength[UnqiueTheadID]; n++) {
				InputNodeID = NeuronInputNodeIDsBuffer[(NeuronNodeStartBuffer[UnqiueTheadID] + n)];
				if (TypeOfNode[UnqiueTheadID] == 0) {
					NodeBufferValue += (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
				}else if (TypeOfNode[UnqiueTheadID] == 1) {
					NodeBufferValue += (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					Sigmoid(NodeBufferValue);
				}else if (TypeOfNode[UnqiueTheadID] == 2) {
					NodeBufferValue += (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					tanh(NodeBufferValue);
				}else if (TypeOfNode[UnqiueTheadID] == 4) {
					NodeBufferValue *= (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
				}else if (TypeOfNode[UnqiueTheadID] == 5) {
					if ((NodeBufferValue > BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID]) {
						NodeBufferValue = (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					}
				}else if (TypeOfNode[UnqiueTheadID] == 6) {
					if ((NodeBufferValue < BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID]) {
						NodeBufferValue = (BufferNodeValues[InputNodeID] * NeuronWeightsBuffer[InputNodeID]) + NeuronBiasBuffer[InputNodeID];
					}
				}

			}

			NeuronOutputNodeBuffer[UnqiueTheadID] = NodeBufferValue;
		}
		
		__syncthreads();
		if (UnqiueTheadID < NumOfNeurons) {			//Transfer the value buffers
			BufferNodeValues[NeuronOutputNodeIDsBuffer[UnqiueTheadID]] = NeuronOutputNodeBuffer[UnqiueTheadID];
		}

		AIScoringFunction(
			UnqiueTheadID,
			NumOfNeurons,
			NeuronOutputNodeBuffer,
			ScoringVariables,
			NetworkScore[0]
		);
		__syncthreads();
		
		//Copy outputNodes data to outputs
		if (UnqiueTheadID < NeuralNetworkSetup[2]) {
			if (OutputIncrementElementEveryXNumOfCylce == 0) {
				OutputData[OutputDataStartArrayID] = BufferNodeValues[OutputNodeID];
			}else if (OutputDataElementSize < i / InputIncrementElementEveryXNumOfCylce) {
				OutputData[OutputDataStartArrayID + (i / OutputIncrementElementEveryXNumOfCylce)] = BufferNodeValues[OutputNodeID];
			}else {
				OutputData[OutputDataStartArrayID + OutputDataElementSize] = BufferNodeValues[OutputNodeID];
			}
		}
		__syncthreads();
		
	}
	//
	NetworkScore[0] *= 0.0001; //Divide score by 10000 to prevent score becoming too large in Core AI thread
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

__global__ void OLD_ALICECoreKernel(
	float *AudioInputBuffer,//I,f,FRAMES_PER_BUFFER
	float *TimeDateInput,
	unsigned int NumOfBufferNodes,
	float *BufferNodeValues,							//IO,f,NumOfBufferNodes
	unsigned int NumOfLSTMUnits,
	unsigned int *LSTMInputNodeIDBuffer,//I,ui,NumOfLSTMUnits
	unsigned int *LSTMOutputNodeIDBuffer,//I,ui,NumOfLSTMUnits
	float *LSTMWeightsListBuffer,						//I,f,NumOfLSTMUnits * 8
	float *LSTMPreviousCellStateValueBuffer,	//ct-1	//IO,f,NumOfLSTMUnits
	float *LSTMPreviousOutputValueBuffer,		//ht-1	//IO,f,NumOfLSTMUnits
	float *LSTMForgetGateValueBuffer,			//ft	//IO,f,NumOfLSTMUnits
	float *LSTMInputGateValueBuffer,			//it	//IO,f,NumOfLSTMUnits
	float *LSTMCandidateValueBuffer,			//~ct	//IO,f,NumOfLSTMUnits
	float *LSTMOutputGateValueBuffer,			//ot	//IO,f,NumOfLSTMUnits
	float *LSTMCellStateValueBuffer,			//ct	//IO,f,NumOfLSTMUnits
	float *LSTMOutputValueBuffer,				//ht	//IO,f,NumOfLSTMUnits
	unsigned int NumOfNeurons,
	unsigned int *NeuronNodeStartBuffer,//I,ui,NumOfNeurons
	unsigned int *NeuronNodeEndBuffer,//I,ui,NumOfNeurons
	unsigned int *NeuronInputNodeIDsBuffer,//I,ui,NumOfNeurons
	unsigned int *NeuronOutputNodeIDsBuffer,
	float *NeuronOutputNodeBuffer,						//I,f,NumOfNeurons
	float *NeuronWeightsBuffer,//I,f,NumOfNeuronWeights
	float *ScoringVariables,//I,f,NumOfScoringVariables
	unsigned int TrainingMode,		//0=Emotinal training, 1=Offline trainer training , 2=Offline training, 3=Online training
	float *NetworkScore,
	float *AudioOutputBuffer)//O,f,FRAMES_PER_BUFFER
{

	unsigned int UnqiueTheadID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ThreadID = threadIdx.x;
	unsigned int BlockID = blockIdx.x;
	unsigned int Blockdim = blockDim.x;
	unsigned int LSTMID = UnqiueTheadID / 3;
	int NeuronID = UnqiueTheadID - (LSTMID * 3);

	float LSTMInputValue;							//xt
	float LSTMPreviousCellStateValue;				//ct-1
	float LSTMPreviousOutputValue;					//ht-1
	
	//Weights
	float LSTMForgetGateInputWeight;			//wf
	float LSTMForgetGatePreviousInputWeight;	//uf
	float LSTMInputGateInputWeight;				//wi
	float LSTMInputGatePreviousInputWeight;		//ui
	float LSTMOutputGateInputWeight;			//wo
	float LSTMOutputGatePreviousInputWeight;	//uo
	float LSTMCandidateGateInputWeight;			//wc
	float LSTMCandidateGatePreviousInputWeight; //uc
	if (LSTMID < NumOfLSTMUnits) {
		LSTMForgetGateInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 0];			//wf
		LSTMForgetGatePreviousInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 1];	//uf
		LSTMInputGateInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 2];				//wi
		LSTMInputGatePreviousInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 3];		//ui
		LSTMOutputGateInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 4];			//wo
		LSTMOutputGatePreviousInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 5];	//uo
		LSTMCandidateGateInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 6];			//wc
		LSTMCandidateGatePreviousInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 7]; //uc
	}
	//End Weights
	//Biases
	float LSTMFogetGateBias = 5;
	float LSTMInputGateBias = 5;
	float LSTMOutputGateBias = 5;
	float LSTMCandidateGateBias = 5;
	//End Biases
	
	float ScoreBuffer = 0;
	//(LSTMID * 3) == UnqiueTheadID


	NetworkScore[0] = 0;
	for (unsigned int i = 0; i < FRAMES_PER_BUFFER; i++) {
		__syncthreads();
		ScoreBuffer = 0;
		if (UnqiueTheadID == 0) {
			BufferNodeValues[0] = AudioInputBuffer[i];
		}else if (UnqiueTheadID <= 5) {//1=sec,2=min,3=hor,4=day,5=mnt
			BufferNodeValues[UnqiueTheadID] = TimeDateInput[UnqiueTheadID - 1];
		}
		
		__syncthreads();
		//Take inputs from GPU buffer to thread local buffers
		if (LSTMID < NumOfLSTMUnits) {
			LSTMInputValue = BufferNodeValues[LSTMInputNodeIDBuffer[LSTMID]];
			//BufferNodeValues[1] = BufferNodeValues[LSTMInputNodeIDBuffer[LSTMID]];
			LSTMPreviousOutputValue = LSTMPreviousOutputValueBuffer[LSTMID];
			LSTMPreviousCellStateValue = LSTMPreviousCellStateValueBuffer[LSTMID];
			if (((LSTMID * 3) + 0) == UnqiueTheadID) {
				RunLSTMUnitStageOneForgetGate(
					LSTMInputValue,							//xt
					LSTMPreviousOutputValue,				//ht-1
					//Weights
					LSTMForgetGateInputWeight,				//wf
					LSTMForgetGatePreviousInputWeight,		//uf
					//End Weights
					//Biases
					LSTMFogetGateBias,
					//End biases
					LSTMForgetGateValueBuffer[LSTMID]		//ft
				);
			}else if (((LSTMID * 3) + 1) == UnqiueTheadID) {
				RunLSTMUnitStageOneInputGate(
					LSTMInputValue,							//xt
					LSTMPreviousOutputValue,				//ht-1
					//Weights
					LSTMInputGateInputWeight,				//wi
					LSTMInputGatePreviousInputWeight,		//ui
					//End Weights
					//Biases
					LSTMInputGateBias,
					//End biases
					LSTMInputGateValueBuffer[LSTMID]		//it
				);
			}else if (((LSTMID * 3) + 2) == UnqiueTheadID) {
				RunLSTMUnitStageOneCandidateGate(
					LSTMInputValue,							//xt
					LSTMPreviousOutputValue,				//ht-1
					//Weights
					LSTMCandidateGateInputWeight,			//wc
					LSTMCandidateGatePreviousInputWeight,	//uc
					//End Weights
					//Biases
					LSTMCandidateGateBias,
					//End biases
					LSTMCandidateValueBuffer[LSTMID]		//~ct
				);
			}
		}


		__syncthreads();
		if (LSTMID < NumOfLSTMUnits) {
			if (((LSTMID * 3) + 0) == UnqiueTheadID) {
				RunLSTMUnitStageTwoOutputGate(
					LSTMInputValue,						//xt
					LSTMPreviousOutputValue,			//ht-1
					//Weights
					LSTMOutputGateInputWeight,			//wo
					LSTMOutputGatePreviousInputWeight,	//uo
					//End Weights
					//Biases
					LSTMOutputGateBias,
					//End Biases
					LSTMOutputGateValueBuffer[LSTMID]	//ot
				);
			}else if (((LSTMID * 3) + 1) == UnqiueTheadID) {
				RunLSTMUnitStageTwoCellState(
					LSTMPreviousCellStateValue,	//ct-1
					LSTMForgetGateValueBuffer[LSTMID],	//ft
					LSTMInputGateValueBuffer[LSTMID],	//it
					LSTMCandidateValueBuffer[LSTMID],	//~ct

					LSTMCellStateValueBuffer[LSTMID]	//ct
				);
			}

		}else if (UnqiueTheadID < (LSTMID * 3) + NumOfNeurons) {
			unsigned int NumOfInputs = NeuronNodeEndBuffer[NeuronID] - NeuronNodeStartBuffer[NeuronID];
			float Activation = 0;
			for (unsigned int n = 0; n > NumOfInputs; n++) {
				Activation += BufferNodeValues[NeuronInputNodeIDsBuffer[n + NeuronNodeStartBuffer[NeuronID]]] * NeuronWeightsBuffer[n + NeuronNodeStartBuffer[NeuronID]];
			}
			Activation = Sigmoid(Activation);
			NeuronOutputNodeBuffer[NeuronID] = Activation;
		}

		__syncthreads();
		if (LSTMID < NumOfLSTMUnits) {
			if (((LSTMID * 3) + 0) == UnqiueTheadID) {
				RunLSTMUnitStageThree(
					LSTMOutputGateValueBuffer[LSTMID],	//ot
					LSTMCellStateValueBuffer[LSTMID],	//ct
					LSTMOutputValueBuffer[LSTMID]		//ht
				);
			}
		}
		__syncthreads();
		//Take data from local thread buffer to GPU buffer
		if (LSTMID < NumOfLSTMUnits) {
			if (((LSTMID * 3) + 0) == UnqiueTheadID) {
				BufferNodeValues[LSTMOutputNodeIDBuffer[LSTMID]] = LSTMOutputValueBuffer[LSTMID];
			//	BufferNodeValues[LSTMOutputNodeIDBuffer[LSTMID]] = 0.7;
			}else if (((LSTMID * 3) + 1) == UnqiueTheadID) {
				LSTMPreviousCellStateValueBuffer[LSTMID] = LSTMCellStateValueBuffer[LSTMID];
			}else  if (((LSTMID * 3) + 2) == UnqiueTheadID) {
				LSTMPreviousOutputValueBuffer[LSTMID] = LSTMOutputValueBuffer[LSTMID];
			}
		}else if(UnqiueTheadID < (NumOfLSTMUnits * 3) + NumOfNeurons){
			BufferNodeValues[NeuronOutputNodeIDsBuffer[NeuronID]] = NeuronOutputNodeBuffer[NeuronID];
		//	BufferNodeValues[(NumOfLSTMUnits * 3) + NeuronID] = 0.7;
		}else if (UnqiueTheadID == (NumOfLSTMUnits * 3) + NumOfNeurons) {
			//0=Emotinal training, 1=Offline trainer training , 2=Offline training, 3=Online training
			//Score the output
			if (TrainingMode == 0) {		//0=Emotinal training
				for (unsigned int n = 0; n < 11; n++) {
					ScoreBuffer = (LSTMOutputValueBuffer[n + 11] - ScoringVariables[n]);
					if (ScoreBuffer > 0) {
						ScoreBuffer = -ScoreBuffer;
					}
					NetworkScore[0] += ScoreBuffer + 1.0;
				}
			}else if (TrainingMode == 1) {	//1=Offline trainer training
				for (unsigned int n = 0; n < 8; n++) {
					ScoreBuffer = (LSTMOutputValueBuffer[n + 8] - ScoringVariables[n]) / 2.0;
					if (ScoreBuffer > 0) {
						ScoreBuffer = -ScoreBuffer;
					}
					NetworkScore[0] += ScoreBuffer + 1.0;
				}

			}else if (TrainingMode == 2) {	//2=Offline training
				for (unsigned int n = 0; n < 8; n++) {
					ScoreBuffer = (LSTMOutputValueBuffer[n + 8] - ScoringVariables[n]) / 2.0;
					if (ScoreBuffer > 0) {
						ScoreBuffer = -ScoreBuffer;
					}
					NetworkScore[0] += ScoreBuffer + 1.0;
				}

			}else if (TrainingMode == 3) {	//3 = Online training
				float Ecstasy = LSTMOutputValueBuffer[11];
				float Admiration = LSTMOutputValueBuffer[12];
				float Terror = LSTMOutputValueBuffer[13];
				float Amazement = LSTMOutputValueBuffer[14];
				float Grief = LSTMOutputValueBuffer[15];
				float Loathing = LSTMOutputValueBuffer[16];
				float Rage = LSTMOutputValueBuffer[17];
				float Vigilance = LSTMOutputValueBuffer[18];
				float ID = LSTMOutputValueBuffer[19];
				float Busy = LSTMOutputValueBuffer[20];
				float Present = LSTMOutputValueBuffer[21];
				//float AdminPresent = ((ID + 1.0) / 2.0) * ((Present + 1.0) / 2.0);
				float AdminPresent = ((0.375 * ID) + 0.625) * ((Present + 1.0) / 2.0);
				NetworkScore[0] += Ecstasy * AdminPresent * 3.0;		//3
				NetworkScore[0] += Admiration * AdminPresent * 4.0;	//4
				NetworkScore[0] += Terror * AdminPresent * -4.0;		//-4
				NetworkScore[0] += Amazement * AdminPresent * 2.0;		//2
				NetworkScore[0] += Grief * AdminPresent * -1.0;		//-1
				NetworkScore[0] += Loathing * AdminPresent;			//-2
				NetworkScore[0] += Rage * AdminPresent * -3.0;			//-3 
				NetworkScore[0] += Vigilance * AdminPresent * 1.0;		//1


				NetworkScore[0] += 0;
			}
		}
		__syncthreads();
	}
	//
	NetworkScore[0] *= 0.0001; //Divide score by 10000 to prevent score becoming too large in Core AI thread
	
}


cudaError_t OLD_RunALICECoreOnGPU(
	float *AudioInputBuffer,/////
	float *TimeDateInput,
	unsigned int NumOfBufferNodes,
	float *BufferNodeValues,
	unsigned int NumOfLSTMUnits,
	unsigned int *LSTMInputNodeIDBuffer,/////
	unsigned int *LSTMOutputNodeIDBuffer,/////
	float *LSTMWeightsListBuffer,/////
	float *LSTMPreviousCellStateValueBuffer,	//ct-1
	float *LSTMPreviousOutputValueBuffer,		//ht-1
	float *LSTMForgetGateValueBuffer,			//ft
	float *LSTMInputGateValueBuffer,			//it
	float *LSTMCandidateValueBuffer,			//~ct
	float *LSTMOutputGateValueBuffer,			//ot
	float *LSTMCellStateValueBuffer,			//ct
	float *LSTMOutputValueBuffer,				//ht
	unsigned int NumOfNeurons,
	unsigned int NumOfNeuronInputs,
	unsigned int *NeuronNodeStartBuffer,//const/////
	unsigned int *NeuronNodeEndBuffer,/////
	unsigned int *NeuronInputNodeIDsBuffer,/////
	unsigned int *NeuronOutputNodeIDsBuffer,
	float *NeuronOutputNodeBuffer,/////
	unsigned int NumOfNeuronWeights,
	float *NeuronWeightsBuffer,/////
	unsigned int NumOfScoringVariables,
	float *ScoringVariables,/////
	unsigned int TrainingMode,		//0=Emotinal training, 1=Offline trainer training , 2=Offline training, 3=Online training
	float *NetworkScore,
	float *AudioOutputBuffer
)
{

	float *dev_AudioInputBuffer = 0;
	float *dev_TimeDateInput = 0;
	unsigned int dev_NumOfBufferNodes = 0;
	float *dev_BufferNodeValues = 0;
	unsigned int dev_NumOfLSTMUnits = 0;
	unsigned int *dev_LSTMInputNodeIDBuffer = 0;
	unsigned int *dev_LSTMOutputNodeIDBuffer = 0;
	float *dev_LSTMWeightsListBuffer = 0;
	float *dev_LSTMPreviousCellStateValueBuffer = 0;	//ct-1
	float *dev_LSTMPreviousOutputValueBuffer = 0;		//ht-1
	float *dev_LSTMForgetGateValueBuffer = 0;			//ft
	float *dev_LSTMInputGateValueBuffer = 0;			//it
	float *dev_LSTMCandidateValueBuffer = 0;			//~ct
	float *dev_LSTMOutputGateValueBuffer = 0;			//ot
	float *dev_LSTMCellStateValueBuffer = 0;			//ct
	float *dev_LSTMOutputValueBuffer = 0;				//ht
	unsigned int dev_NumOfNeurons = 0;
	unsigned int dev_NumOfNeuronInputs = 0;
	unsigned int *dev_NeuronNodeStartBuffer = 0;
	unsigned int *dev_NeuronNodeEndBuffer = 0;
	unsigned int *dev_NeuronInputNodeIDsBuffer = 0;
	unsigned int *dev_NeuronOutputNodeIDsBuffer = 0;
	float *dev_NeuronOutputNodeBuffer = 0;
	float *dev_NeuronWeightsBuffer = 0;
	float *dev_ScoringVariables = 0;
	unsigned int dev_TrainingMode = 0;		//0=Emotinal training, 1=Offline training, 2=Online training
	float *dev_NetworkScore = 0;
	float *dev_AudioOutputBuffer = 0;
	float *dev_EmotinoalOutputs = 0;
	cudaError_t cudaStatus;

	dev_NumOfBufferNodes = NumOfBufferNodes;
	dev_NumOfLSTMUnits = NumOfLSTMUnits;
	dev_NumOfNeurons = NumOfNeurons;
	dev_NumOfNeuronInputs = NumOfNeuronInputs;
	dev_TrainingMode = TrainingMode;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_AudioInputBuffer, NUM_OF_TIME_VARIABLES * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_AudioInputBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_TimeDateInput, FRAMES_PER_BUFFER * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_TimeDateInput");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_BufferNodeValues, dev_NumOfBufferNodes * sizeof(float));
	//cudaStatus = cudaMalloc((void**)&dev_BufferNodeValues, 33 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_BufferNodeValues");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMInputNodeIDBuffer, dev_NumOfLSTMUnits * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMInputNodeIDBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMOutputNodeIDBuffer, dev_NumOfLSTMUnits * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMOutputNodeIDBuffer");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_LSTMWeightsListBuffer, dev_NumOfLSTMUnits * 8 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMWeightsListBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMPreviousCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMPreviousCellStateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMPreviousOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMPreviousOutputValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMForgetGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMForgetGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMInputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMInputGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMCandidateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMCandidateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMOutputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMOutputGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMCellStateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMOutputValueBuffer");
		goto Error;
	}



	cudaStatus = cudaMalloc((void**)&dev_NeuronNodeStartBuffer, dev_NumOfNeurons * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_NeuronInputNodeStartBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NeuronNodeEndBuffer, dev_NumOfNeurons * sizeof(unsigned int));
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
	cudaStatus = cudaMalloc((void**)&dev_ScoringVariables, NumOfScoringVariables * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_ScoringVariables");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NetworkScore, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_AudioOutputBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_AudioOutputBuffer, FRAMES_PER_BUFFER * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_AudioOutputBuffer");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_AudioInputBuffer, AudioInputBuffer, FRAMES_PER_BUFFER * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_AudioInputBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_TimeDateInput, TimeDateInput, NUM_OF_TIME_VARIABLES * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_TimeDateInput");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_BufferNodeValues, BufferNodeValues, dev_NumOfBufferNodes * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_BufferNodeValues");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_LSTMInputNodeIDBuffer, LSTMInputNodeIDBuffer, dev_NumOfLSTMUnits * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMInputNodeIDBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMOutputNodeIDBuffer, LSTMOutputNodeIDBuffer, dev_NumOfLSTMUnits * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMOutputNodeIDBuffer");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_LSTMWeightsListBuffer, LSTMWeightsListBuffer, dev_NumOfLSTMUnits * 8 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMWeightsListBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMPreviousCellStateValueBuffer, LSTMPreviousCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMPreviousCellStateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMPreviousOutputValueBuffer, LSTMPreviousOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMPreviousOutputValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMForgetGateValueBuffer, LSTMForgetGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMForgetGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMInputGateValueBuffer, LSTMInputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMInputGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMCandidateValueBuffer, LSTMCandidateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMCandidateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMOutputGateValueBuffer, LSTMOutputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMOutputGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMCellStateValueBuffer, LSTMCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMCellStateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMOutputValueBuffer, LSTMOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMOutputValueBuffer");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_NeuronNodeStartBuffer, NeuronNodeStartBuffer, dev_NumOfNeurons * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_NeuronInputNodeStartBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_NeuronNodeEndBuffer, NeuronNodeEndBuffer, dev_NumOfNeurons * sizeof(unsigned int), cudaMemcpyHostToDevice);
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
	cudaStatus = cudaMemcpy(dev_ScoringVariables, ScoringVariables, NumOfScoringVariables * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_ScoringVariables");
		goto Error;
	}




	// Launch a kernel on the GPU with one thread for each element.
	OLD_ALICECoreKernel <<<1,256>>>(
		dev_AudioInputBuffer,
		dev_TimeDateInput,
		dev_NumOfBufferNodes,
		dev_BufferNodeValues,
		dev_NumOfLSTMUnits,
		dev_LSTMInputNodeIDBuffer,
		dev_LSTMOutputNodeIDBuffer,
		dev_LSTMWeightsListBuffer,
		dev_LSTMPreviousCellStateValueBuffer,	//ct-1
		dev_LSTMPreviousOutputValueBuffer,		//ht-1
		dev_LSTMForgetGateValueBuffer,			//ft
		dev_LSTMInputGateValueBuffer,			//it
		dev_LSTMCandidateValueBuffer,			//~ct
		dev_LSTMOutputGateValueBuffer,			//ot
		dev_LSTMCellStateValueBuffer,			//ct
		dev_LSTMOutputValueBuffer,				//ht
		dev_NumOfNeurons,
		dev_NeuronNodeStartBuffer,
		dev_NeuronNodeEndBuffer,
		dev_NeuronInputNodeIDsBuffer,
		dev_NeuronOutputNodeIDsBuffer,
		dev_NeuronOutputNodeBuffer,
		dev_NeuronWeightsBuffer,
		dev_ScoringVariables,
		dev_TrainingMode,		//0=Emotinal training, 1=Offline training, 2=Online training
		dev_NetworkScore,
		dev_AudioOutputBuffer
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
	cudaStatus = cudaMemcpy(BufferNodeValues, dev_BufferNodeValues, dev_NumOfBufferNodes * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(LSTMPreviousCellStateValueBuffer, dev_LSTMPreviousCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMPreviousOutputValueBuffer, dev_LSTMPreviousOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMForgetGateValueBuffer, dev_LSTMForgetGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMInputGateValueBuffer, dev_LSTMInputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMCandidateValueBuffer, dev_LSTMCandidateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMOutputGateValueBuffer, dev_LSTMOutputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMCellStateValueBuffer, dev_LSTMCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMOutputValueBuffer, dev_LSTMOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(NetworkScore, dev_NetworkScore, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(AudioOutputBuffer, dev_AudioOutputBuffer, FRAMES_PER_BUFFER * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_AudioInputBuffer);
	cudaFree(dev_TimeDateInput);
	cudaFree(dev_BufferNodeValues);
	cudaFree(dev_LSTMInputNodeIDBuffer);
	cudaFree(dev_LSTMOutputNodeIDBuffer);
	cudaFree(dev_LSTMWeightsListBuffer);
	cudaFree(dev_LSTMPreviousCellStateValueBuffer);
	cudaFree(dev_LSTMPreviousOutputValueBuffer);
	cudaFree(dev_LSTMForgetGateValueBuffer);
	cudaFree(dev_LSTMInputGateValueBuffer);
	cudaFree(dev_LSTMCandidateValueBuffer);
	cudaFree(dev_LSTMOutputGateValueBuffer);
	cudaFree(dev_LSTMCellStateValueBuffer);
	cudaFree(dev_LSTMOutputValueBuffer);
	cudaFree(dev_NeuronNodeStartBuffer);
	cudaFree(dev_NeuronNodeEndBuffer);
	cudaFree(dev_NeuronInputNodeIDsBuffer);
	cudaFree(dev_NeuronOutputNodeIDsBuffer);
	cudaFree(dev_NeuronOutputNodeBuffer);
	cudaFree(dev_NeuronWeightsBuffer);
	cudaFree(dev_ScoringVariables);
	cudaFree(dev_AudioOutputBuffer);
	cudaFree(dev_EmotinoalOutputs);

	return cudaStatus;
}



int OLD_RunALICECore(
	vector<float>& AudioInputBuffer,
	vector<float>& TimeDateInput,
	vector<float>& BufferNodeValues,
	vector<unsigned int>& LSTMInputNodeIDBuffer,
	vector<unsigned int>& LSTMOutputNodeIDBuffer,
	vector<float>& LSTMWeightsListBuffer,
	vector<float>& LSTMPreviousCellStateValueBuffer,	//ct-1
	vector<float>& LSTMPreviousOutputValueBuffer,		//ht-1
	vector<float>& LSTMForgetGateValueBuffer,			//ft
	vector<float>& LSTMInputGateValueBuffer,			//it
	vector<float>& LSTMCandidateValueBuffer,			//~ct
	vector<float>& LSTMOutputGateValueBuffer,			//ot
	vector<float>& LSTMCellStateValueBuffer,			//ct
	vector<float>& LSTMOutputValueBuffer,				//ht
	vector<unsigned int>& NeuronNodeStartBuffer,
	vector<unsigned int>& NeuronNodeEndBuffer,
	vector<unsigned int>& NeuronInputNodeIDsBuffer,
	vector<unsigned int>& NeuronOutputNodeIDsBuffer,
	vector<float>& NeuronOutputNodeBuffer,
	vector<float>& NeuronWeightsBuffer,
	vector<float>& ScoringVariables,
	unsigned int TrainingMode,		//0=Emotinal training, 1=Offline trainer training , 2=Offline training, 3=Online training
	float& NetworkScore,
	vector<float>& AudioOutputBuffer
){
	cudaError_t cudaStatus = OLD_RunALICECoreOnGPU(
		&AudioInputBuffer[0],
		&TimeDateInput[0],
		BufferNodeValues.size(),
		&BufferNodeValues[0],
		LSTMInputNodeIDBuffer.size(),
		&LSTMInputNodeIDBuffer[0],
		&LSTMOutputNodeIDBuffer[0],
		&LSTMWeightsListBuffer[0],
		&LSTMPreviousCellStateValueBuffer[0],	//ct-1
		&LSTMPreviousOutputValueBuffer[0],		//ht-1
		&LSTMForgetGateValueBuffer[0],			//ft
		&LSTMInputGateValueBuffer[0],			//it
		&LSTMCandidateValueBuffer[0],			//~ct
		&LSTMOutputGateValueBuffer[0],			//ot
		&LSTMCellStateValueBuffer[0],			//ct
		&LSTMOutputValueBuffer[0],				//ht
		NeuronOutputNodeBuffer.size(),
		NeuronInputNodeIDsBuffer.size(),
		&NeuronNodeStartBuffer[0],
		&NeuronNodeEndBuffer[0],
		&NeuronInputNodeIDsBuffer[0],
		&NeuronOutputNodeIDsBuffer[0],
		&NeuronOutputNodeBuffer[0],
		NeuronWeightsBuffer.size(),
		&NeuronWeightsBuffer[0],
		ScoringVariables.size(),
		&ScoringVariables[0],
		TrainingMode,		//0=Emotinal training, 1=Offline training, 2=Online training
		&NetworkScore,
		&AudioOutputBuffer[0]
	);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "RunALICECore failed!");
		return 1;
	}

	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
}



__global__ void OLD_ALICECoreNoStandardNeuronsKernel(
	float *AudioInputBuffer,//I,f,FRAMES_PER_BUFFER
	float *TimeDateInput,
	unsigned int NumOfBufferNodes,
	float *BufferNodeValues,							//IO,f,NumOfBufferNodes
	unsigned int NumOfLSTMUnits,
	unsigned int *LSTMInputNodeIDBuffer,//I,ui,NumOfLSTMUnits
	unsigned int *LSTMOutputNodeIDBuffer,//I,ui,NumOfLSTMUnits
	float *LSTMWeightsListBuffer,						//I,f,NumOfLSTMUnits * 8
	float *LSTMPreviousCellStateValueBuffer,	//ct-1	//IO,f,NumOfLSTMUnits
	float *LSTMPreviousOutputValueBuffer,		//ht-1	//IO,f,NumOfLSTMUnits
	float *LSTMForgetGateValueBuffer,			//ft	//IO,f,NumOfLSTMUnits
	float *LSTMInputGateValueBuffer,			//it	//IO,f,NumOfLSTMUnits
	float *LSTMCandidateValueBuffer,			//~ct	//IO,f,NumOfLSTMUnits
	float *LSTMOutputGateValueBuffer,			//ot	//IO,f,NumOfLSTMUnits
	float *LSTMCellStateValueBuffer,			//ct	//IO,f,NumOfLSTMUnits
	float *LSTMOutputValueBuffer,				//ht	//IO,f,NumOfLSTMUnits
	float *ScoringVariables,//I,f,NumOfScoringVariables
	unsigned int TrainingMode,		//0=Emotinal training, 1=Offline trainer training , 2=Offline training, 3=Online training
	float *NetworkScore,
	float *AudioOutputBuffer)//O,f,FRAMES_PER_BUFFER
{

	unsigned int UnqiueTheadID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ThreadID = threadIdx.x;
	unsigned int BlockID = blockIdx.x;
	unsigned int Blockdim = blockDim.x;
	unsigned int LSTMID = UnqiueTheadID / 3;
	int NeuronID = UnqiueTheadID - (LSTMID * 3);

	float LSTMInputValue;							//xt
	float LSTMPreviousCellStateValue;				//ct-1
	float LSTMPreviousOutputValue;					//ht-1
													//Weights
	float LSTMForgetGateInputWeight;			//wf
	float LSTMForgetGatePreviousInputWeight;	//uf
	float LSTMInputGateInputWeight;				//wi
	float LSTMInputGatePreviousInputWeight;		//ui
	float LSTMOutputGateInputWeight;			//wo
	float LSTMOutputGatePreviousInputWeight;	//uo
	float LSTMCandidateGateInputWeight;			//wc
	float LSTMCandidateGatePreviousInputWeight; //uc
	if (LSTMID < NumOfLSTMUnits) {
		LSTMForgetGateInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 0];			//wf
		LSTMForgetGatePreviousInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 1];	//uf
		LSTMInputGateInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 2];				//wi
		LSTMInputGatePreviousInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 3];		//ui
		LSTMOutputGateInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 4];			//wo
		LSTMOutputGatePreviousInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 5];	//uo
		LSTMCandidateGateInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 6];			//wc
		LSTMCandidateGatePreviousInputWeight = LSTMWeightsListBuffer[(LSTMID * 8) + 7]; //uc
	}
	//End Weights
	//Biases
	float LSTMFogetGateBias = 5;
	float LSTMInputGateBias = 5;
	float LSTMOutputGateBias = 5;
	float LSTMCandidateGateBias = 5;
	//End Biases

	float ScoreBuffer = 0;
	//(LSTMID * 3) == UnqiueTheadID


	NetworkScore[0] = 0;
	for (unsigned int i = 0; i < FRAMES_PER_BUFFER; i++) {
		__syncthreads();
		ScoreBuffer = 0;
		if (UnqiueTheadID == 0) {
			BufferNodeValues[0] = AudioInputBuffer[i];
		}else if (UnqiueTheadID <= 5) {
			BufferNodeValues[UnqiueTheadID] = TimeDateInput[UnqiueTheadID - 1];
		}

		__syncthreads();
		//Take inputs from GPU buffer to thread local buffers
		if (LSTMID < NumOfLSTMUnits) {
			LSTMInputValue = BufferNodeValues[LSTMInputNodeIDBuffer[LSTMID]];
			//BufferNodeValues[1] = BufferNodeValues[LSTMInputNodeIDBuffer[LSTMID]];
			LSTMPreviousOutputValue = LSTMPreviousOutputValueBuffer[LSTMID];
			LSTMPreviousCellStateValue = LSTMPreviousCellStateValueBuffer[LSTMID];
			if (((LSTMID * 3) + 0) == UnqiueTheadID) {
				RunLSTMUnitStageOneForgetGate(
					LSTMInputValue,							//xt
					LSTMPreviousOutputValue,				//ht-1
					//Weights
					LSTMForgetGateInputWeight,				//wf
					LSTMForgetGatePreviousInputWeight,		//uf
					//End Weights
					//Biases
					LSTMFogetGateBias,
					//End biases
					LSTMForgetGateValueBuffer[LSTMID]		//ft
				);
			}else if (((LSTMID * 3) + 1) == UnqiueTheadID) {
				RunLSTMUnitStageOneInputGate(
					LSTMInputValue,							//xt
					LSTMPreviousOutputValue,				//ht-1
					//Weights
					LSTMInputGateInputWeight,				//wi
					LSTMInputGatePreviousInputWeight,		//ui
					//End Weights
					//Biases
					LSTMInputGateBias,
					//End biases
					LSTMInputGateValueBuffer[LSTMID]		//it
				);
			}else if (((LSTMID * 3) + 2) == UnqiueTheadID) {
				RunLSTMUnitStageOneCandidateGate(
					LSTMInputValue,							//xt
					LSTMPreviousOutputValue,				//ht-1
					//Weights
					LSTMCandidateGateInputWeight,			//wc
					LSTMCandidateGatePreviousInputWeight,	//uc
					//End Weights
					//Biases
					LSTMCandidateGateBias,
					//End biases
					LSTMCandidateValueBuffer[LSTMID]		//~ct
				);
			}
		}


		__syncthreads();
		if (LSTMID < NumOfLSTMUnits) {
			if (((LSTMID * 3) + 0) == UnqiueTheadID) {
				RunLSTMUnitStageTwoOutputGate(
					LSTMInputValue,						//xt
					LSTMPreviousOutputValue,			//ht-1
					//Weights
					LSTMOutputGateInputWeight,			//wo
					LSTMOutputGatePreviousInputWeight,	//uo
					//End Weights
					//Biases
					LSTMOutputGateBias,
					//End Biases
					LSTMOutputGateValueBuffer[LSTMID]	//ot
				);
			}
			else if (((LSTMID * 3) + 1) == UnqiueTheadID) {
				RunLSTMUnitStageTwoCellState(
					LSTMPreviousCellStateValue,	//ct-1
					LSTMForgetGateValueBuffer[LSTMID],	//ft
					LSTMInputGateValueBuffer[LSTMID],	//it
					LSTMCandidateValueBuffer[LSTMID],	//~ct

					LSTMCellStateValueBuffer[LSTMID]	//ct
				);
			}

		}

		__syncthreads();
		if (LSTMID < NumOfLSTMUnits) {
			if (((LSTMID * 3) + 0) == UnqiueTheadID) {
				RunLSTMUnitStageThree(
					LSTMOutputGateValueBuffer[LSTMID],	//ot
					LSTMCellStateValueBuffer[LSTMID],	//ct
					LSTMOutputValueBuffer[LSTMID]		//ht
				);
			}
		}
		__syncthreads();
		//Take data from local thread buffer to GPU buffer
		if (LSTMID < NumOfLSTMUnits) {
			if (((LSTMID * 3) + 0) == UnqiueTheadID) {
				BufferNodeValues[LSTMOutputNodeIDBuffer[LSTMID]] = LSTMOutputValueBuffer[LSTMID];
				//	BufferNodeValues[LSTMOutputNodeIDBuffer[LSTMID]] = 0.7;
			}else if (((LSTMID * 3) + 1) == UnqiueTheadID) {
				LSTMPreviousCellStateValueBuffer[LSTMID] = LSTMCellStateValueBuffer[LSTMID];
			}else  if (((LSTMID * 3) + 2) == UnqiueTheadID) {
				LSTMPreviousOutputValueBuffer[LSTMID] = LSTMOutputValueBuffer[LSTMID];
			}
		}else if (UnqiueTheadID == (NumOfLSTMUnits * 3)) {
			//Score the output
			if (TrainingMode == 0) {		//0=Emotinal training
				for (unsigned int n = 0; n < 11; n++) {
					ScoreBuffer = (LSTMOutputValueBuffer[n + 11] - ScoringVariables[n]) / 2.0;
					if (ScoreBuffer > 0) {
						ScoreBuffer = -ScoreBuffer;
					}
					NetworkScore[0] += ScoreBuffer + 1.0;
				}
			}else if (TrainingMode == 1) {	//1=Offline trainer training
				for (unsigned int n = 0; n < 8; n++) {
					ScoreBuffer = (LSTMOutputValueBuffer[n + 8] - ScoringVariables[n]) / 2.0;
					if (ScoreBuffer > 0) {
						ScoreBuffer = -ScoreBuffer;
					}
					NetworkScore[0] += ScoreBuffer + 1.0;
				}

			}else if (TrainingMode == 1) {	//2=Offline training
				for (unsigned int n = 0; n < 8; n++) {
					ScoreBuffer = (LSTMOutputValueBuffer[n + 8] - ScoringVariables[n]) / 2.0;
					if (ScoreBuffer > 0) {
						ScoreBuffer = -ScoreBuffer;
					}
					NetworkScore[0] += ScoreBuffer + 1.0;
				}

			}else if (TrainingMode == 2) {	//3 = Online training
				float Ecstasy = LSTMOutputValueBuffer[11];
				float Admiration = LSTMOutputValueBuffer[12];
				float Terror = LSTMOutputValueBuffer[13];
				float Amazement = LSTMOutputValueBuffer[14];
				float Grief = LSTMOutputValueBuffer[15];
				float Loathing = LSTMOutputValueBuffer[16];
				float Rage = LSTMOutputValueBuffer[17];
				float Vigilance = LSTMOutputValueBuffer[18];
				float ID = LSTMOutputValueBuffer[19];
				float Busy = LSTMOutputValueBuffer[20];
				float Present = LSTMOutputValueBuffer[21];
				float AdminPresent = ((ID + 1.0) / 2.0) * ((Present + 1.0) / 2.0);
				NetworkScore[0] += Ecstasy * AdminPresent * 3.0;		//3
				NetworkScore[0] += Admiration * AdminPresent * 4.0;	//4
				NetworkScore[0] += Terror * AdminPresent * -4.0;		//-4
				NetworkScore[0] += Amazement * AdminPresent * 2.0;		//2
				NetworkScore[0] += Grief * AdminPresent * -1.0;		//-1
				NetworkScore[0] += Loathing * AdminPresent;			//-2
				NetworkScore[0] += Rage * AdminPresent * -3.0;			//-3 
				NetworkScore[0] += Vigilance * AdminPresent * 1.0;		//1


				NetworkScore[0] += 0;
			}
		}
		__syncthreads();
	}
	//
	NetworkScore[0] *= 0.0001; //Divide score by 10000 to prevent score becoming too large in Core AI thread
	//
}


cudaError_t OLD_RunALICECoreNoStandardNeuronsOnGPU(
	float *AudioInputBuffer,/////
	float *TimeDateInput,
	unsigned int NumOfBufferNodes,
	float *BufferNodeValues,
	unsigned int NumOfLSTMUnits,
	unsigned int *LSTMInputNodeIDBuffer,/////
	unsigned int *LSTMOutputNodeIDBuffer,/////
	float *LSTMWeightsListBuffer,/////
	float *LSTMPreviousCellStateValueBuffer,	//ct-1
	float *LSTMPreviousOutputValueBuffer,		//ht-1
	float *LSTMForgetGateValueBuffer,			//ft
	float *LSTMInputGateValueBuffer,			//it
	float *LSTMCandidateValueBuffer,			//~ct
	float *LSTMOutputGateValueBuffer,			//ot
	float *LSTMCellStateValueBuffer,			//ct
	float *LSTMOutputValueBuffer,				//ht
	unsigned int NumOfScoringVariables,
	float *ScoringVariables,/////
	unsigned int TrainingMode,		//0=Emotinal training, 1=Offline training, 2=Online training
	float *NetworkScore,
	float *AudioOutputBuffer
)
{

	float *dev_AudioInputBuffer = 0;
	float *dev_TimeDateInput;
	unsigned int dev_NumOfBufferNodes = 0;
	float *dev_BufferNodeValues = 0;
	unsigned int dev_NumOfLSTMUnits = 0;
	unsigned int *dev_LSTMInputNodeIDBuffer = 0;
	unsigned int *dev_LSTMOutputNodeIDBuffer = 0;
	float *dev_LSTMWeightsListBuffer = 0;
	float *dev_LSTMPreviousCellStateValueBuffer = 0;	//ct-1
	float *dev_LSTMPreviousOutputValueBuffer = 0;		//ht-1
	float *dev_LSTMForgetGateValueBuffer = 0;			//ft
	float *dev_LSTMInputGateValueBuffer = 0;			//it
	float *dev_LSTMCandidateValueBuffer = 0;			//~ct
	float *dev_LSTMOutputGateValueBuffer = 0;			//ot
	float *dev_LSTMCellStateValueBuffer = 0;			//ct
	float *dev_LSTMOutputValueBuffer = 0;				//ht
	float *dev_ScoringVariables = 0;
	unsigned int dev_TrainingMode = 0;		//0=Emotinal training, 1=Offline training, 2=Online training
	float *dev_NetworkScore = 0;
	float *dev_AudioOutputBuffer = 0;
	float *dev_EmotinoalOutputs = 0;
	cudaError_t cudaStatus;

	dev_NumOfBufferNodes = NumOfBufferNodes;
	dev_NumOfLSTMUnits = NumOfLSTMUnits;
	dev_TrainingMode = TrainingMode;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_AudioInputBuffer, FRAMES_PER_BUFFER * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_AudioInputBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_TimeDateInput, NUM_OF_TIME_VARIABLES * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_TimeDateInput");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_BufferNodeValues, dev_NumOfBufferNodes * sizeof(float));
	//cudaStatus = cudaMalloc((void**)&dev_BufferNodeValues, 33 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_BufferNodeValues");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMInputNodeIDBuffer, dev_NumOfLSTMUnits * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMInputNodeIDBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMOutputNodeIDBuffer, dev_NumOfLSTMUnits * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMOutputNodeIDBuffer");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_LSTMWeightsListBuffer, dev_NumOfLSTMUnits * 8 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMWeightsListBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMPreviousCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMPreviousCellStateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMPreviousOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMPreviousOutputValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMForgetGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMForgetGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMInputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMInputGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMCandidateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMCandidateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMOutputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMOutputGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMCellStateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_LSTMOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_LSTMOutputValueBuffer");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_ScoringVariables, NumOfScoringVariables * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_ScoringVariables");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_NetworkScore,  sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_AudioOutputBuffer");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_AudioOutputBuffer, FRAMES_PER_BUFFER * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_AudioOutputBuffer");
		goto Error;
	}

	
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_AudioInputBuffer, AudioInputBuffer, FRAMES_PER_BUFFER * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_AudioInputBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_TimeDateInput, TimeDateInput, NUM_OF_TIME_VARIABLES * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_TimeDateInput");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_BufferNodeValues, BufferNodeValues, dev_NumOfBufferNodes * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_BufferNodeValues");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_LSTMInputNodeIDBuffer, LSTMInputNodeIDBuffer, dev_NumOfLSTMUnits * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMInputNodeIDBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMOutputNodeIDBuffer, LSTMOutputNodeIDBuffer, dev_NumOfLSTMUnits * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMOutputNodeIDBuffer");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_LSTMWeightsListBuffer, LSTMWeightsListBuffer, dev_NumOfLSTMUnits * 8 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMWeightsListBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMPreviousCellStateValueBuffer, LSTMPreviousCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMPreviousCellStateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMPreviousOutputValueBuffer, LSTMPreviousOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMPreviousOutputValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMForgetGateValueBuffer, LSTMForgetGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMForgetGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMInputGateValueBuffer, LSTMInputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMInputGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMCandidateValueBuffer, LSTMCandidateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMCandidateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMOutputGateValueBuffer, LSTMOutputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMOutputGateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMCellStateValueBuffer, LSTMCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMCellStateValueBuffer");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_LSTMOutputValueBuffer, LSTMOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_LSTMOutputValueBuffer");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_ScoringVariables, ScoringVariables, NumOfScoringVariables * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! dev_ScoringVariables");
		goto Error;
	}




	// Launch a kernel on the GPU with one thread for each element.
	OLD_ALICECoreNoStandardNeuronsKernel << <1, 256 >> >(
		dev_AudioInputBuffer,
		dev_TimeDateInput,
		dev_NumOfBufferNodes,
		dev_BufferNodeValues,
		dev_NumOfLSTMUnits,
		dev_LSTMInputNodeIDBuffer,
		dev_LSTMOutputNodeIDBuffer,
		dev_LSTMWeightsListBuffer,
		dev_LSTMPreviousCellStateValueBuffer,	//ct-1
		dev_LSTMPreviousOutputValueBuffer,		//ht-1
		dev_LSTMForgetGateValueBuffer,			//ft
		dev_LSTMInputGateValueBuffer,			//it
		dev_LSTMCandidateValueBuffer,			//~ct
		dev_LSTMOutputGateValueBuffer,			//ot
		dev_LSTMCellStateValueBuffer,			//ct
		dev_LSTMOutputValueBuffer,				//ht
		dev_ScoringVariables,
		dev_TrainingMode,		//0=Emotinal training, 1=Offline training, 2=Online training
		dev_NetworkScore,
		dev_AudioOutputBuffer
		);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ALICECoreNoStandardNeuronsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(BufferNodeValues, dev_BufferNodeValues, dev_NumOfBufferNodes * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(LSTMPreviousCellStateValueBuffer, dev_LSTMPreviousCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMPreviousOutputValueBuffer, dev_LSTMPreviousOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMForgetGateValueBuffer, dev_LSTMForgetGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMInputGateValueBuffer, dev_LSTMInputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMCandidateValueBuffer, dev_LSTMCandidateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMOutputGateValueBuffer, dev_LSTMOutputGateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMCellStateValueBuffer, dev_LSTMCellStateValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(LSTMOutputValueBuffer, dev_LSTMOutputValueBuffer, dev_NumOfLSTMUnits * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(NetworkScore, dev_NetworkScore, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(AudioOutputBuffer, dev_AudioOutputBuffer, FRAMES_PER_BUFFER * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_AudioInputBuffer);
	cudaFree(dev_TimeDateInput);
	cudaFree(dev_BufferNodeValues);
	cudaFree(dev_LSTMInputNodeIDBuffer);
	cudaFree(dev_LSTMOutputNodeIDBuffer);
	cudaFree(dev_LSTMWeightsListBuffer);
	cudaFree(dev_LSTMPreviousCellStateValueBuffer);
	cudaFree(dev_LSTMPreviousOutputValueBuffer);
	cudaFree(dev_LSTMForgetGateValueBuffer);
	cudaFree(dev_LSTMInputGateValueBuffer);
	cudaFree(dev_LSTMCandidateValueBuffer);
	cudaFree(dev_LSTMOutputGateValueBuffer);
	cudaFree(dev_LSTMCellStateValueBuffer);
	cudaFree(dev_LSTMOutputValueBuffer);
	cudaFree(dev_ScoringVariables);
	cudaFree(dev_AudioOutputBuffer);
	cudaFree(dev_EmotinoalOutputs);

	return cudaStatus;
}



int OLD_RunALICECoreNoStandardNeurons(
	vector<float>& AudioInputBuffer,
	vector<float>& TimeDateInput,
	vector<float>& BufferNodeValues,
	vector<unsigned int>& LSTMInputNodeIDBuffer,
	vector<unsigned int>& LSTMOutputNodeIDBuffer,
	vector<float>& LSTMWeightsListBuffer,
	vector<float>& LSTMPreviousCellStateValueBuffer,	//ct-1
	vector<float>& LSTMPreviousOutputValueBuffer,		//ht-1
	vector<float>& LSTMForgetGateValueBuffer,			//ft
	vector<float>& LSTMInputGateValueBuffer,			//it
	vector<float>& LSTMCandidateValueBuffer,			//~ct
	vector<float>& LSTMOutputGateValueBuffer,			//ot
	vector<float>& LSTMCellStateValueBuffer,			//ct
	vector<float>& LSTMOutputValueBuffer,				//ht
	vector<float>& ScoringVariables,
	unsigned int TrainingMode,		//0=Emotinal training, 1=Offline training, 2=Online training
	float &NetworkScore,
	vector<float>& AudioOutputBuffer
) {
	cudaError_t cudaStatus = OLD_RunALICECoreNoStandardNeuronsOnGPU(
		&AudioInputBuffer[0],
		&TimeDateInput[0],
		BufferNodeValues.size(),
		&BufferNodeValues[0],
		LSTMInputNodeIDBuffer.size(),
		&LSTMInputNodeIDBuffer[0],
		&LSTMOutputNodeIDBuffer[0],
		&LSTMWeightsListBuffer[0],
		&LSTMPreviousCellStateValueBuffer[0],	//ct-1
		&LSTMPreviousOutputValueBuffer[0],		//ht-1
		&LSTMForgetGateValueBuffer[0],			//ft
		&LSTMInputGateValueBuffer[0],			//it
		&LSTMCandidateValueBuffer[0],			//~ct
		&LSTMOutputGateValueBuffer[0],			//ot
		&LSTMCellStateValueBuffer[0],			//ct
		&LSTMOutputValueBuffer[0],				//ht
		ScoringVariables.size(),
		&ScoringVariables[0],
		TrainingMode,		//0=Emotinal training, 1=Offline training, 2=Online training
		&NetworkScore,
		&AudioOutputBuffer[0]
	);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "RunALICECoreNoStandardNeurons failed!");
		return 1;
	}

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
}

