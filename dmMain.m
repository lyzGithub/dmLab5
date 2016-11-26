%implimentation for AdaBoost algorithm( Naive Bayes as base)
%by liyize 2016 11 21
%http://lamda.nju.edu.cn/yehj/DM16/dm16.html
%?P(C|F1F2...Fn) 
%         = P(F1F2...Fn|C)P(C) / P(F1F2...Fn)
%    P(F1F2...Fn|C)P(C) 
%         = P(F1|C)P(F2|C) ... P(Fn|C)P(C)
% 1  representing discrete feature and 0 representing numerical feature
%data1: 1-9 all is discrete feature: [0,10]
%data2: 1-12  numerical feature:[0,1]  13-24 discrete feature:(0 or 1)
clear;
data1Path = 'data/breast-cancer-assignment5.txt';
data2Path = 'data/german-assignment5.txt';
data1Matrix = csvread(data1Path);
data2Matrix = csvread(data2Path);
[m1,n1]  =size(data1Matrix);
[m2,n2]  =size(data2Matrix);
% feature1 = data1Matrix(1,:) ;
% feature2 = data2Matrix(1,:) ;
% feature1(n1) = [];
% feature2(n2) = [];
data1Matrix(1,:) = [];
data2Matrix(1,:) = [];
data1Label = data1Matrix(:,n1);
data2Label = data2Matrix(:,n2);
data1Matrix(:,n1) = [];
data2Matrix(:,n2) = [];
maxf = max(data1Matrix);
%first imple naive bayes
 disp('data1');
%data1
% naiveBayes1( trainData, testData, trainLabel, testLabel )
 result11 = naiveBayes1(data1Matrix,data1Matrix,data1Label,data1Label);
 naive1Rate = result11.accRate;

%adaboost for data1
%ten fold for build
sev1 = floor(m1/10);
numToGo = 1;
mean1 = zeros(numToGo,1);
allS1 = zeros(numToGo,1);
numToGoVector = zeros(numToGo,1);
for mki = 1:numToGo
    numToGoVector(mki) = mki;
    mean1(mki) = 0;
    for i = 1:10
        %myAdaboost1( trainData, trainLabel,  testData, testLabel,M )
        %testData array
        testBeginIndex = 1+sev1*(i-1);
        testEndindex = sev1 * i;
        testData = data1Matrix((testBeginIndex:testEndindex),:);
        testLabel = data1Label((testBeginIndex:testEndindex),:);
        %train data array
        trainData = data1Matrix;
        trainLabel = data1Label;
        trainData((testBeginIndex:testEndindex),:) = [];
        trainLabel((testBeginIndex:testEndindex),:) = [];
        result12(i) = myAdaboost1( trainData, trainLabel,  testData, testLabel,  10);
        mean1(mki) = mean1(mki) +result12(i).accRate;
    end
    %
    allS1(mki) = 0;
    for i = 1:10
        allS1(mki) = allS1(mki) + (result12(i).accRate - mean1(mki))^2;
    end
    allS1(mki) = sqrt(allS1(mki));
   % disp(['accuracy1 mean1:',num2str(mean1(mki)),', standard deviation1:',num2str(allS1(mki))]);
end
figure('NumberTitle', 'off', 'Name', 'data1mean1')
plot(numToGoVector, mean1,'r');
grid on;
xlabel('time of iteration');  
ylabel('mean1Correct');
legend('mean1');


figure('NumberTitle', 'off', 'Name', 'data1standard deviation')
plot(numToGoVector,allS1,'g');
grid on;
xlabel('time of iteration');  
ylabel('standard deviation');
legend('standard deviation');






disp('data2');
 %data2
 result21 = naiveBayes2(data2Matrix,data2Matrix,data2Label,data2Label);
 naive2Rate = result21.accRate;
 %adaboost for data2
%ten fold
sev2 = floor(m2/10);
numToGo = 1;
mean2 = zeros(numToGo,1);
allS2 = zeros(numToGo,1);
numToGoVector = zeros(numToGo,1);
for mki = 1:numToGo
    numToGoVector(mki) = mki;
    mean2(mki) = 0;
    for i = 1:10
        testBeginIndex = 1+sev2*(i-1);
        testEndindex = sev2 * i;
        testData = data2Matrix((testBeginIndex:testEndindex),:);
        testLabel = data2Label((testBeginIndex:testEndindex),:);
        %train data array
        trainData = data2Matrix;
        trainLabel = data2Label;
        trainData((testBeginIndex:testEndindex),:) = [];
        trainLabel((testBeginIndex:testEndindex),:) = [];
        result22(i) = myAdaboost2( trainData, trainLabel,  testData, testLabel,  10);
        mean2(mki) = mean2(mki) +result22(i).accRate;
    end
    %
    allS2(mki) = 0;
    for i = 1:10
        allS2(mki) = allS2(mki) + (result22(i).accRate - mean2(mki))^2;
    end
    allS2(mki) = sqrt(allS2(mki));
    %disp(['accuracy2 mean2:',num2str(mean2),', standard deviation2:',num2str(allS2)]);
    
end
figure('NumberTitle', 'off', 'Name', 'data2mean2')
plot(numToGoVector, mean2,'r');
grid on;
xlabel('time of iteration');  
ylabel('mean1Correct');
legend('mean1');

figure('NumberTitle', 'off', 'Name', 'data2 standard deviation')
plot(numToGoVector,allS2,'g');
grid on;
xlabel('time of iteration');  
ylabel('standard deviation');
legend('standard deviation');
 
 
 
 