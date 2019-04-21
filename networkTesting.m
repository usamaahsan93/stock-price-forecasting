%% Network Testing Part

%Opening sample test data
load net
fileID = fopen('Dataset\Microsoft_Test.csv');
% fileID = fopen('Dataset\FB.csv');
% fileID = fopen('Dataset\GOLD.csv');
% fileID = fopen('Dataset\DVMT.csv');
fgetl(fileID);
C = textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);

% Formatting the Validation Data
Open = cell2mat(C(1, 2));
Open = Open';
High = cell2mat(C(1, 3));
High = High';
Low = cell2mat(C(1, 4));
Low = Low';
Close = cell2mat(C(1, 5));
Close = Close';

% Taking Simple and Moving Average
simpleMovAvg_10 = tsmovavg(Open, 's', 10);
simpleMovAvg_50 = tsmovavg(Open, 's', 50);
expMovAvg_10 = tsmovavg(Open, 'e', 10);
expMovAvg_50 = tsmovavg(Open, 'e', 50);

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

% Testing Network
Input = [Open; High; Low; simpleMovAvg_10; expMovAvg_10; simpleMovAvg_50; expMovAvg_50];
answer = ones(1, size(Close, 2));

for i=1:size(Close, 2)
        answer(i)=net(Input(:,i));
end

% Calculating Performance Error
retMSE=mse(answer,Close);

% Plotting Output Graph 
x = 1:size(Close, 2);
axis tight
plot(x, answer, x, Close);
legend('Predicted Value','Actual Value','Location','southeast')
xlabel('Data Points');
ylabel('Closing Stock Market Value');
title('Stock Market Prediction Validation using Neural Networks');
grid on;
