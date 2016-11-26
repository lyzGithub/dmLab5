function [ result] = myAdaboost1( trainData, trainLabel,  testData, testLabel,M )
%M : for iter times
%preModel: am weight for every classfier
[d,n] = size(trainData);
dW = zeros(1,d);
dW(:) = 1/d;
selectSize = [d,1];%rand size

i  = 1;
while i <= M
  
    selectIndex = discretize(rand(selectSize), [0 cumsum(dW)]);
    trainDatai = trainData(selectIndex,:);
    trainLabeli = trainLabel(selectIndex);
    resultStruct(i) = naiveBayes1(trainDatai,trainDatai,trainLabeli,trainLabeli);
    if resultStruct(i).accRate < 0.5
        continue;
    end
    
    %renew the weight for every sample if it is predicted correct
    ant = (resultStruct(i).preLabel ==trainLabeli);
    renewIndex = find(ant == 1);% correct class detect
    oldWeightSum = sum(dW);
    errorRate = 1-resultStruct(i).accRate;
    for j = 1:length(renewIndex)
        dW(renewIndex(j)) =  dW(renewIndex(j)) * (errorRate/(1-errorRate));
    end
    % reduce to 1
    newWeightSum = sum(dW);
    dW = dW *(oldWeightSum/newWeightSum);
    i = i+1;
end

% testData for predict for adaboost
preLabel = zeros(size(testLabel));
[mtTe,nTe]  = size(testData);

for i = 1:mtTe
    vTemp = testData(i,:);
    %pro for class0
    weigthTwo = zeros(2,1);
    for mi = 1:M
        errorRate = 1- resultStruct(mi).accRate;
        wj = log((1-errorRate)/errorRate);
        F = resultStruct(mi).F;
        Cp =  resultStruct(mi).Cp;
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
        weigthTwo(1 ) =  weigthTwo(1 )  + pro0 * wj;
        weigthTwo(2 ) =  weigthTwo(2 )  + pro1 * wj;
    end
    if weigthTwo(2 ) >= weigthTwo(1 )
        preLabel(i) = 1;
    elseif weigthTwo(2 ) < weigthTwo(1 )
        preLabel(i)  = 0;
    end
    
end

ant = (preLabel ==testLabel);
result.accRate = length(find(ant == 1))/mtTe;
result.preLabel = preLabel;

end

