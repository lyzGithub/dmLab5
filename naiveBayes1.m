function [ result ] = naiveBayes1( trainData, testData, trainLabel, testLabel )
% implement for naive Bayes
%?P(C|F1F2...Fn)
%         = P(F1F2...Fn|C)P(C) / P(F1F2...Fn)
%    P(F1F2...Fn|C)P(C)
%         = P(F1|C)P(F2|C) ... P(Fn|C)P(C)
%feature for number type? %data1: 1-9 all is discrete feature: [0,10]
[mTr,~]  =size(trainData);
[mtTe,nTe]  =size(testData);
%compute tow class pro
% learn from train, Ci pro,  P(F1|C)
Cp.c1 = length(find(trainLabel==1))/mTr;
Cp.c0 = 1 - Cp.c1;

%feature length: 9, feature :0-10 F(ai/ci),F.fi
F = proFi1(trainData,trainLabel);%F :F = zeros(2,11*n);
%test
preLabel = zeros(size(testLabel));
for i = 1:mtTe
    vTemp = testData(i,:);
    %pro for class0
    pro0 = Cp.c0;
    for j = 1:nTe
        %F(1, int32(1+  vTemp(j) +(j-1)*11) )
        pro0 = pro0  *  F(1, int32(1+  vTemp(j) +(j-1)*11) );
    end
    %pro for class1
    pro1 = Cp.c1;
    for j = 1:nTe
        pro1 = pro1  *  F(2, int32(1+ vTemp(j)+(j-1)*11) );
    end
    %pro1
    %pro0
    if pro1 >= pro0
        preLabel(i) = 1;
    elseif pro1 < pro0
        preLabel(i)  = 0;
    end
end
%accurace
ant = (preLabel ==testLabel);
result.accRate = length(find(ant == 1))/mtTe;
result.preLabel = preLabel;
result.F = F;
result.Cp = Cp;
end

