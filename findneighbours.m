function Nbs = findneighbours(CMA,POP,iter,maxIter,BI,NN)
index = (1/(1+log(1.6718-iter/maxIter))-0.6605)*BI.l_dim/BI.u_dim;
dst = dist(reshape([POP.UX],CMA.lambda,BI.u_dim))+index*dist(reshape([POP.LX],CMA.lambda,BI.l_dim));
[value,ind]  = sort(dst,2);
for i = 1 : CMA.lambda  
    Nbs(i).ind = ind(i,1:NN+1);
    Nbs(i).value = value(i,1:NN+1); 
end    

function dis = dist(val)
[m,n] = size(val);
dis = zeros(m,n);
for i = 1: m
    for j = i + 1 : m    
        dis(i,j)= (sum((abs(val(i,:)-val(j,:))).^(1/n)))^(n);
    end
end
dis = dis +dis';