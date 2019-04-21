%% Stock Price Prediction Project By Usama Ahsan

%% Network Training Part

%% Parameters determined by Genetic Algorithm
% Optimal Parameters found
epoch=9000;
learningRate=0.01;
goal=6e-05;
neuron=10;

%% Reading Data

fileID = fopen('Dataset\Microsoft_Train.csv');
fgetl(fileID); 
% delimitting the various sub-fields
C=textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);

%% Formatting Data

% Opening stock value for the day
Open = cell2mat(C(1,2));
Open = Open';

% Highest stock value for the day
High = cell2mat(C(1,3));
High = High';

% Lowest stock value for the day
Low = cell2mat(C(1,4));
Low = Low';

% Closing stock value for the day
Close = cell2mat(C(1,5));
Close = Close';

%% Taking Simple and Moving Average

% Simple Moving Average for 10 and 50 days
simpleMovAvg_10 = tsmovavg(Open,'s',10);
simpleMovAvg_50 = tsmovavg(Open,'s',50);

% Exponential Moving Average for 10 and 50 days
expMovAvg_10 = tsmovavg(Open,'e',10);
expMovAvg_50 = tsmovavg(Open,'e',50);

% Limiting the vector '50:end' because starting values returned after
% moving average is NaN, because of unavailability of data before 1
expMovAvg_10=expMovAvg_10(1,50:end);
expMovAvg_50=expMovAvg_50(1,50:end);
simpleMovAvg_10=simpleMovAvg_10(1,50:end);
simpleMovAvg_50=simpleMovAvg_50(1,50:end);
Open=Open(1,50:end);
High=High(1,50:end);
Low=Low(1,50:end);
Close=Close(1,50:end);

%% Setting up the Neural Network

% Input vector of the input variables
Input = [Open; High; Low; simpleMovAvg_10; expMovAvg_10; simpleMovAvg_50; expMovAvg_50];

net=feedforwardnet(neuron,'traingdx');
net.layers{1}.transferFcn = 'purelin';
net.divideFcn ='dividetrain';

% Setting the Parameters
net.trainparam.epochs = epoch;
net.trainparam.goal =goal;
net.trainparam.lr = learningRate;

net = train(net, Input, Close);


%% Validation Test of the constructed neural network
%%

% Opening sample test data
fileID = fopen('Dataset\Microsoft_Validation.csv');
fgetl(fileID);
C_t = textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);

% Formatting the Validation Data
openVal = cell2mat(C_t(1, 2));
openVal = openVal';
highVal = cell2mat(C_t(1, 3));
highVal = highVal';
LowVal = cell2mat(C_t(1, 4));
LowVal = LowVal';
closeVal = cell2mat(C_t(1, 5));
closeVal = closeVal';

% Taking Simple and Moving Average
simpleMovAvg_10_Val = tsmovavg(openVal, 's', 10);
simpleMovAvg_50_Val = tsmovavg(openVal, 's', 50);
expMovAvg_10_Val = tsmovavg(openVal, 'e', 10);
expMovAvg_50_Val = tsmovavg(openVal, 'e', 50);

% Limiting the vector '50:end' because starting values returned after
% moving average is NaN, because of unavailability of data before 1
expMovAvg_10_Val=expMovAvg_10_Val(1,50:end);
expMovAvg_50_Val=expMovAvg_50_Val(1,50:end);
simpleMovAvg_10_Val=simpleMovAvg_10_Val(1,50:end);
simpleMovAvg_50_Val=simpleMovAvg_50_Val(1,50:end);
openVal=openVal(1,50:end);
highVal=highVal(1,50:end);
LowVal=LowVal(1,50:end);
closeVal=closeVal(1,50:end);

% Validating Network
Input_t = [openVal; highVal; LowVal; simpleMovAvg_10_Val; expMovAvg_10_Val; simpleMovAvg_50_Val; expMovAvg_50_Val];
answer = ones(1, size(closeVal, 2));

for i=1:size(closeVal, 2)
        answer(i)=net(Input_t(:,i));
end

% Calculating Performance Error
retMSE=mse(answer,closeVal);

% Plotting Output
x = 1:size(closeVal, 2);
axis tight
plot(x, answer, x, closeVal);
legend('Predicted Value','Actual Value','Location','southeast')
xlabel('Data Points');
ylabel('Closing Stock Market Value');
title('Stock Market Prediction Validation using Neural Network');
grid on

save net