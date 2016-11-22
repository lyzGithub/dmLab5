function [ result ] = naiveBayes2( trainData, testData, trainLabel, testLabel )
% implement for naive Bayes
%?P(C|F1F2...Fn) 
%         = P(F1F2...Fn|C)P(C) / P(F1F2...Fn)
%    P(F1F2...Fn|C)P(C) 
%         = P(F1|C)P(F2|C) ... P(Fn|C)P(C)
%feature for number type
[mTr,nTr]  =size(trainData);
[mtTe,nTe]  =size(testData);


result.accRate = 0;
result.preLabel = 0;

end