function [ result ] = naiveBayes2( trainData, testData, trainLabel, testLabel )
% implement for naive Bayes
%?P(C|F1F2...Fn)
%         = P(F1F2...Fn|C)P(C) / P(F1F2...Fn)
%    P(F1F2...Fn|C)P(C)
%         = P(F1|C)P(F2|C) ... P(Fn|C)P(C)
%feature for number type,%data2: 1-12  numerical feature:[0,1]  13-24 discrete feature:(0 or 1)
%features:[0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0],
[mTr,~]  =size(trainData);
[mtTe,nTe]  =size(testData);
Cp.c1 = length(find(trainLabel==1))/mTr;
Cp.c0 = 1 - Cp.c1;
F = proFi2(trainData,trainLabel);%F :F = zeros(2,11*n);

%test
preLabel = zeros(size(testLabel));
for i = 1:mtTe
    vTemp = testData(i,:);
    %pro for class0
    pro0 = Cp.c0;
    for j = 1:nTe
        tempInt1 = floor(vTemp(j) * 5 +1);
        if tempInt1 >5
            tempInt1 = 5;
        end
        pro0 = pro0  *  F(1, int32(tempInt1 +(j-1)*5) );
    end
    %pro for class1
    pro1 = Cp.c1;
    for j = 1:nTe
        tempInt2 = floor(vTemp(j) * 5 +1);
        if tempInt2 >5
            tempInt2 = 5;
        end
        pro1 = pro1  *  F(2, int32( tempInt2+(j-1)*5) );
    end
    %    pro1
    %    pro0
    if pro1 >= pro0
        preLabel(i) = 1;
    elseif pro1 < pro0
        preLabel(i)  = -1;
    end
end
%accurace
ant = (preLabel ==testLabel);
result.accRate = length(find(ant == 1))/mtTe;
result.preLabel = preLabel;
result.F = F;
result.Cp = Cp;
end