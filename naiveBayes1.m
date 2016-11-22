function [ result ] = naiveBayes1( trainData, testData, trainLabel, testLabel )
% implement for naive Bayes
%?P(C|F1F2...Fn) 
%         = P(F1F2...Fn|C)P(C) / P(F1F2...Fn)
%    P(F1F2...Fn|C)P(C) 
%         = P(F1|C)P(F2|C) ... P(Fn|C)P(C)
%feature for number type? %data1: 1-9 all is discrete feature: [0,10]
[mTr,nTr]  =size(trainData);
[mtTe,nTe]  =size(testData);
%compute tow class pro
C.c1Pro;
C.c2Pro;




result.accRate = 0;
result.preLabel = 0;

end

