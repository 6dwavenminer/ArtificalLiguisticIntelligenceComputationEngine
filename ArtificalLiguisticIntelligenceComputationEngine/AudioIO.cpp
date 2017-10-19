#include "AudioIO.h"


int AudioIOThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	unsigned int TrainingMode,
	atomic<unsigned int>& AudioIOToggledUpdateFromMicFlag,
	atomic<unsigned int>& AudioIOToggledUpdateToSpeakerFlag,
	vector<float>& AudioInputBufferFromMicThread,
	vector<float> *AudioInputBufferToSpeakerThread
) {
	if (TrainingMode != 0) {
		PaStreamParameters inputParameters, outputParameters;
		PaStream *stream = NULL;
		PaError err;
		const PaDeviceInfo* inputInfo;
		const PaDeviceInfo* outputInfo;
		float buffer[FRAMES_PER_BUFFER * 2][2];
		int i;
		int numBytes;
		int numChannels;
		vector<float> VecFloat;
		vector<float> AudioBufferFromMic;
		vector<float> AudioBufferToSpeaker;

		printf("patest_read_write_wire.c\n"); fflush(stdout);
		printf("sizeof(int) = %lu\n", sizeof(int)); fflush(stdout);
		printf("sizeof(long) = %lu\n", sizeof(long)); fflush(stdout);

		err = Pa_Initialize();
		if (err != paNoError) goto error2;

		inputParameters.device = Pa_GetDefaultInputDevice(); // default input device 
		printf("Input device # %d.\n", inputParameters.device);
		inputInfo = Pa_GetDeviceInfo(inputParameters.device);
		printf("    Name: %s\n", inputInfo->name);
		printf("      LL: %g s\n", inputInfo->defaultLowInputLatency);
		printf("      HL: %g s\n", inputInfo->defaultHighInputLatency);

		outputParameters.device = Pa_GetDefaultOutputDevice(); // default output device 
		printf("Output device # %d.\n", outputParameters.device);
		outputInfo = Pa_GetDeviceInfo(outputParameters.device);
		printf("   Name: %s\n", outputInfo->name);
		printf("     LL: %g s\n", outputInfo->defaultLowOutputLatency);
		printf("     HL: %g s\n", outputInfo->defaultHighOutputLatency);

		numChannels = inputInfo->maxInputChannels < outputInfo->maxOutputChannels
			? inputInfo->maxInputChannels : outputInfo->maxOutputChannels;
		printf("Num channels = %d.\n", numChannels);

		inputParameters.channelCount = numChannels;
		inputParameters.sampleFormat = PA_SAMPLE_TYPE;
		inputParameters.suggestedLatency = inputInfo->defaultHighInputLatency;
		inputParameters.hostApiSpecificStreamInfo = NULL;

		outputParameters.channelCount = numChannels;
		outputParameters.sampleFormat = PA_SAMPLE_TYPE;
		outputParameters.suggestedLatency = outputInfo->defaultHighOutputLatency;
		outputParameters.hostApiSpecificStreamInfo = NULL;

		// -- setup -- //

		err = Pa_OpenStream(
			&stream,
			&inputParameters,
			&outputParameters,
			SAMPLE_RATE,
			FRAMES_PER_BUFFER,
			paClipOff,      // we won't output out of range samples so don't bother clipping them //
			NULL, // no callback, use blocking API //
			NULL); // no callback, so no callback userData //
		if (err != paNoError) goto error2;

		numBytes = FRAMES_PER_BUFFER * numChannels * SAMPLE_SIZE;

		err = Pa_StartStream(stream);
		if (err != paNoError) goto error1;
		printf("Wire on. Will run %d seconds.\n", NUM_SECONDS); fflush(stdout);
		while (*ProgramStartup == 1) {
			//Wait for startup to finish
		}
		while (*RunProgram == 1) {
			//for (i = 0; i < (NUM_SECONDS*SAMPLE_RATE) / FRAMES_PER_BUFFER; ++i)
			//{
				// You may get underruns or overruns if the output is not primed by PortAudio.
			err = Pa_ReadStream(stream, buffer, FRAMES_PER_BUFFER);
			/////if (err) goto xrun;
			AudioBufferFromMic.clear();
			for (int j = 0; j < FRAMES_PER_BUFFER; j++) {
				AudioBufferFromMic.push_back(buffer[j][0]);
			}

			AudioInputBufferFromMicThread.resize(AudioBufferFromMic.size());
			copy(AudioBufferFromMic.begin(), AudioBufferFromMic.end(), AudioInputBufferFromMicThread.begin());
			AudioIOToggledUpdateFromMicFlag ^= 1;

			if ((*AudioInputBufferToSpeakerThread).size() == FRAMES_PER_BUFFER) {
				AudioBufferToSpeaker.resize(FRAMES_PER_BUFFER);
				copy((*AudioInputBufferToSpeakerThread).begin(), (*AudioInputBufferToSpeakerThread).begin() + FRAMES_PER_BUFFER, AudioBufferToSpeaker.begin());
			}
			AudioIOToggledUpdateToSpeakerFlag ^= 1;

			if (AudioBufferToSpeaker.size() == FRAMES_PER_BUFFER) {
				for (int j = 0; j < FRAMES_PER_BUFFER; j++) {
					buffer[j][0] = AudioBufferToSpeaker[j] * 1;
					buffer[j][1] = AudioBufferToSpeaker[j] * 1;
				}
			}else {
				for (int j = 0; j < FRAMES_PER_BUFFER; j++) {
					buffer[j][0] = 0.0;
					buffer[j][1] = 0.0;
				}
			}
			err = Pa_WriteStream(stream, buffer, FRAMES_PER_BUFFER);
			////if (err) goto xrun;
		}
		err = Pa_StopStream(stream);

		if (err != paNoError) goto error1;

		//free(sampleBlock);

		Pa_Terminate();

		return 0;

	xrun:
		printf("err = %d\n", err); fflush(stdout);
		if (stream) {
			Pa_AbortStream(stream);
			Pa_CloseStream(stream);

		}
		//free(sampleBlock);
		Pa_Terminate();
		if (err & paInputOverflow)
			fprintf(stderr, "Input Overflow.\n");
		if (err & paOutputUnderflow)
			fprintf(stderr, "Output Underflow.\n");
		return -2;
	error1:
		//free(sampleBlock);
	error2:
		if (stream) {
			Pa_AbortStream(stream);
			Pa_CloseStream(stream);

		}
		Pa_Terminate();
		fprintf(stderr, "An error occured while using the portaudio stream\n");
		fprintf(stderr, "Error number: %d\n", err);
		fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
	}
	return 0;
}
