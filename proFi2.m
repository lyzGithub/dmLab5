function [ F ] = proFi2( trainData,trainLabel )
%%data2: 1-12  numerical feature:[0,1]  13-24 discrete feature:(0 or 1)
%1-12features:[0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0],,place
%all feature in different cluster
label1Locate = find(trainLabel == 1);
label0Locate = find(trainLabel == -1);
[~,n] = size(trainData);
numC1 = length(label1Locate);
numC0 = length(label0Locate);

F = zeros(2,5*n);

%label0Locate
for i = 1:n
    prov = zeros(1,5);
    %[0,0.2)
    prov(1) = length( find(  trainData(label0Locate, i)>=0 & trainData(label0Locate, i)  < 0.2 )  );
    prov(2) = length( find(   trainData(label0Locate, i)>=0.2 & trainData(label0Locate, i) <0.4 )  );
    prov(3) = length( find(   trainData(label0Locate, i)>=0.4 & trainData(label0Locate, i) <0.6 )  );
    prov(4) = length( find(   trainData(label0Locate, i)>=0.6 & trainData(label0Locate, i) <0.8 )  );
    prov(5) = length( find(   trainData(label0Locate, i)>=0.8 & trainData(label0Locate, i) <=1.0 )  );
    
    for j = 1:5
        prov(j) = (prov(j)+1)/(numC0+5);
    end
    
    F(1,(1+(i-1)*5):(5+(i-1)*5)) = prov;
end
%label1Locate
for i = 1:n
    prov = zeros(1,5);
    %[0,0.2)
    prov(1) = length( find(  trainData(label1Locate, i)>=0 & trainData(label1Locate, i)  < 0.2 )  );
    prov(2) = length( find(   trainData(label1Locate, i)>=0.2 & trainData(label1Locate, i) <0.4 )  );
    prov(3) = length( find(   trainData(label1Locate, i)>=0.4 & trainData(label1Locate, i) <0.6 )  );
    prov(4) = length( find(   trainData(label1Locate, i)>=0.6 & trainData(label1Locate, i) <0.8 )  );
    prov(5) = length( find(   trainData(label1Locate, i)>=0.8 & trainData(label1Locate, i) <=1.0 )  );
    for j = 1:5
        prov(j) = (prov(j)+1)/(numC1+5);
    end
    F(2,(1+(i-1)*5):(5+(i-1)*5)) = prov;
end

end

