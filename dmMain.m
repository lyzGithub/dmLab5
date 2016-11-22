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
feature1 = data1Matrix(1,:) ;
feature2 = data2Matrix(1,:) ;
% feature1(n1) = [];
% feature2(n2) = [];
% data1Matrix(1,:) = [];
% data2Matrix(1,:) = [];
data1Label = data1Matrix(:,n1);
data2Label = data2Matrix(:,n2);
data1Matrix(:,n1) = [];
data2Matrix(:,n2) = [];

%first imple naive bayes
%data1
result1 = naiveBayes1(data1Matrix,data1Matrix,data1Label,data1Label);
 %data2
result2 = naiveBayes2(data2Matrix,data2Matrix,data2Label,data2Label);
 
 
 
 
 
 