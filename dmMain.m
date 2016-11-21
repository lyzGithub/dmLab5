%implimentation for AdaBoost algorithm
%by liyize 2016 11 21
%http://lamda.nju.edu.cn/yehj/DM16/dm16.html
%?P(C|F1F2...Fn) 
%         = P(F1F2...Fn|C)P(C) / P(F1F2...Fn)
%    P(F1F2...Fn|C)P(C) 
%         = P(F1|C)P(F2|C) ... P(Fn|C)P(C)
% 1  representing discrete feature and 0 representing numerical feature
%data1: all is discrete feature [0,10]
%data2: 1-12  numerical feature[0,1]  13-24 discrete feature:(0 or 1)
clear;
data1Path = 'data/breast-cancer-assignment5.txt';
data2Path = 'data/german-assignment5.txt';

data1Matrix = csvread(data1Path);
data2Matrix = csvread(data2Path);
 
 
 
 
 
 
 
 