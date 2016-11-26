function [ F ] = proFi1( trainData,trainLabel )
%feature length: 9, feature :0-10 F(ai/ci),F.fPVi
label1Locate = find(trainLabel == 1);
label0Locate = find(trainLabel == 0);

numC1 = length(label1Locate);
numC0 = length(label0Locate);
[~,n] = size(trainData);
F = zeros(2,11*n);
%class0
for i = 1:n
    prov = zeros(1,11);
    for j = 1:numC0
        switch trainData(label0Locate(j), i)
            case 0
                prov (1) =  prov (1) +1;
            case 1
                prov (2) =  prov (2) +1;
            case 2
                prov (3) =  prov (3) +1;
            case 3
                prov (4) =  prov (4) +1;
            case 4
                prov (5) =  prov (5) +1;
            case 5
                prov (6) =  prov (6) +1;
            case 6
                prov (7) =  prov (7) +1;
            case 7
                prov (8) =  prov (8) +1;
            case 8
                prov (9) =  prov (9) +1;
            case 9
                prov (10) =  prov (10) +1;
            otherwise
                prov (11) =  prov (11) +1;
                
        end
    end
    %prov
    for j = 1:11
        prov(j) = (prov(j)+1)/(numC0+11);%laplas crrect
    end
      %prov
     %disp('~~~~');
    F(1,(1+(i-1)*11):(11+(i-1)*11)) = prov;
end

%class1
for i = 1:n
    prov = zeros(1,11);
    for j = 1:numC1
        switch trainData(label1Locate(j), i)
            case 0
                prov (1) =  prov (1) +1;
            case 1
                prov (2) =  prov (2) +1;
            case 2
                prov (3) =  prov (3) +1;
            case 3
                prov (4) =  prov (4) +1;
            case 4
                prov (5) =  prov (5) +1;
            case 5
                prov (6) =  prov (6) +1;
            case 6
                prov (7) =  prov (7) +1;
            case 7
                prov (8) =  prov (8) +1;
            case 8
                prov (9) =  prov (9) +1;
            case 9
                prov (10) =  prov (10) +1;
            otherwise
                prov (11) =  prov (11) +1;
        end
    end
    for j = 1:11
        prov(j) = (prov(j)+1)/(numC1+11);
    end
    F(2,(1+(i-1)*11):(11+(i-1)*11)) = prov;
end

end

