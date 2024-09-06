% Download pretrained network
downloadFolder = matlab.internal.examples.downloadSupportFile("audio","speechEnhancement/FaSNet.zip");
dataFolder = tempdir;
unzip(downloadFolder,dataFolder)
netFolder = fullfile(dataFolder,"speechEnhancement");
addpath(netFolder)
% Load and Inspect Data
[cleanSpeech,fs] = audioread("cleanSpeech.wav");

soundsc(cleanSpeech,fs)
[ambisonicData,fs] = audioread("ambisonicRecording.wav");
channel = W; # Change accordingly X,Y,Z
soundsc(ambisonicData(:,channel),fs)
compareAudio(cleanSpeech,ambisonicData,SampleRate=fs)
compareSpectrograms(cleanSpeech,ambisonicData)
compareSpectrograms(cleanSpeech,ambisonicData,Warp="mel")

% Perform 3D Speech Enhancement
model = seModel(netFolder);
enhancedSpeech = enhanceSpeech(model,ambisonicData);
soundSource = enhancedSpeech; % Clean Speech, Noisy Ambisonic Recording (W)
soundsc(soundSource,fs)
% Compare the clean speech, noisy speech, and enhanced speech in the time domain, as spectrograms, and as mel spectrograms.
compareAudio(cleanSpeech,ambisonicData,enhancedSpeech)
compareSpectrograms(cleanSpeech,ambisonicData,enhancedSpeech)
compareSpectrograms(cleanSpeech,ambisonicData,enhancedSpeech,Warp="mel")

%Speech Enhancement for Speech-to-Text Applications
%Compare the performance of the speech enhancement system on a downstream speech-to-text system. Use the wav2vec 2.0 speech-to-text model. This model requires a one-time download of pretrained weights to run. If you have not downloaded the wav2vec weights, the first call to speechClient will provide a download link.

%Create the wav2vec 2.0 speech client to perform transcription.

transcriber = speechClient("wav2vec2.0",segmentation="none");
%Perform speech-to-text transcription using the clean speech, the ambisonic data, and the enhanced speech.

cleanSpeechResults = speech2text(transcriber,cleanSpeech,fs)
cleanSpeechResults = "i tell you it is not poison she cried"
noisySpeechResults = speech2text(transcriber,ambisonicData(:,channel),fs)
noisySpeechResults = "i tell you it is not parzona she cried"
enhancedSpeechResults = speech2text(transcriber,enhancedSpeech,fs)
enhancedSpeechResults = "i tell you it is not poisen she cried"













































