 
function ins = TLEACMAES(BI)
global ulFunctionEvaluations;
global llFunctionEvaluations;
ulFunctionEvaluations = 0;
llFunctionEvaluations = 0;
ins.runPath = pwd;
ins.BI = BI;
ins.BI.fn = strvcat(ins.BI.fn);
ins.alg = 'TLEACMAES';

%% parameter settings
elite = [];
record = [];
CMA = initCMAES(BI);
maxIter = ceil(BI.UmaxFEs/CMA.lambda);
ImprIter = ceil(BI.UmaxImprFEs/CMA.lambda);
for iter = 1 : maxIter  
    for i = 1 : CMA.lambda
        U = CMA.xmean + CMA.sigma * randn(1,BI.dim) .* CMA.D * CMA.B';
        flag = U > BI.xrange(2,:);
        U(flag) = (CMA.xmean(flag) + BI.xrange(2,flag))/2;
        flag = U < BI.xrange(1,:);
        U(flag) = (CMA.xmean(flag) + BI.xrange(1,flag))/2;
        POP(i).UX = U(1:BI.u_dim);
        POP(i).LX = U(BI.u_dim+1:end);    
    end  
%% parallel lower level search
    POP = parallelLLS(POP,CMA,BI); 
%% upper level function evaluation
    for i = 1 : CMA.lambda
        [POP(i).UF,POP(i).UC] = UL_evaluate(POP(i).UX,POP(i).LX,BI.fn);
        POP(i).UFEs = ulFunctionEvaluations;
        POP(i).LFEs = llFunctionEvaluations;
    end    
    POP = assignUpperFitness(POP,BI);
    RF_idx = find([POP.RF]);
    if isempty(RF_idx) 
        RF_idx = 1:length(POP); 
    end
    [~,bestIdx] = max([POP(RF_idx).fit]);
    bestIdx = RF_idx(bestIdx);
    bestIndv = POP(bestIdx);

    % replace the elite
    if upperLevelComparator(bestIndv,elite,BI) || rand > 0.5
    	bestIndv = refine(bestIndv,CMA,BI);
    	POP(bestIdx) = bestIndv;
    	if upperLevelComparator(bestIndv,elite,BI)
    		elite = bestIndv;
    	end
    else
    	elite = refine(elite,CMA,BI);
    end
    % re-assign fitness
    POP = assignUpperFitness(POP,BI);
    % recording
    elite.UFEs = ulFunctionEvaluations;    
    elite.LFEs = llFunctionEvaluations;
    record = [record;elite];

%% upper level termination check
    reachTheTarget_b = abs(bestIndv.UF-BI.u_fopt) < BI.u_ftol;
    reachTheTarget_e = abs(elite.UF-BI.u_fopt) < BI.u_ftol;
    reachTheMaxFEs = ulFunctionEvaluations >= BI.UmaxFEs;
    reachTheFlat = iter>ImprIter && abs(record(iter).UF-record(iter-ImprIter+1).UF)/(abs(record(1).UF)+abs(record(iter).UF)) < BI.u_ftol; 
    reachTheFlat = reachTheFlat && abs(record(iter).UF-record(iter-ImprIter+1).UF) < BI.u_ftol;
    
    if reachTheTarget_b || reachTheTarget_e || reachTheMaxFEs || reachTheFlat
        if reachTheTarget_b
            elite = bestIndv;
        end
        break;
    end
    CMA = updateCMAES(CMA,POP,BI);
end

ins.UF = elite.UF;
ins.LF = elite.LF;
ins.UX = elite.UX;
ins.LX = elite.LX;
ins.UFEs = ulFunctionEvaluations;
ins.LFEs = llFunctionEvaluations;
ins.record = record;

%% transfer learning based parallel lower level CMA-ES search
function POP = parallelLLS(POP,CMA,BI)   
    lambda = 4+floor(3*log(BI.l_dim));
    maxIter  = ceil(BI.LmaxFEs / lambda);
    ImprIter = ceil(BI.LmaxImprFEs /lambda);

    for i = 1 : CMA.lambda     
        termination(i) = false;
        Lrecord(i,:) = zeros(1,maxIter);      
        LCMA(i) = initLCMAES(CMA,BI,POP(i).LX);
        POP(i).RF = false; 
    end
       sigma0 = 1;
 for iter = 1 :maxIter
      for i = 1 : CMA.lambda
         if ~termination(i)
             LCMA(i)= OnestepLS(POP(i).UX,LCMA(i),BI);
             POP(i).LX = LCMA(i).bestIndv.LX;
             POP(i).LC = LCMA(i).bestIndv.LC;
             POP(i).LF = LCMA(i).bestIndv.LF;
             Lrecord(i,iter) = LCMA(i).bestIndv.LF;            
            if iter>ImprIter
                if abs(Lrecord(i,iter)-Lrecord(i,iter-2))/(abs(Lrecord(i,1))+abs(Lrecord(i,iter))) < 1e-4
                    LCMA(i).TR = 1;
                else
                    LCMA(i).TR = 0;
                end
                LCMA(i).LR = abs(Lrecord(i,iter)-Lrecord(i,iter-ImprIter+1))/(abs(Lrecord(i,1))+abs(Lrecord(i,iter)));              
            end            
            if (iter>ImprIter && abs(Lrecord(i,iter)-Lrecord(i,iter-ImprIter+1))/(abs(Lrecord(i,1))+abs(Lrecord(i,iter))) < 1e-4) ...
                    || (iter > ImprIter && abs(Lrecord(i,iter) - Lrecord(i,iter-ImprIter+1)) < 10*BI.l_ftol) ...
                    || LCMA(i).sigma / sigma0 < 1e-2 ...
                    || LCMA(i).sigma / sigma0 > 1e2         
                termination(i) = true;
                POP(i).RF = true;     
            end   
         end         
      end
      if  sum(termination)==CMA.lambda
          break;       
      end
    
     if  iter>ImprIter&& mod(iter,1)==0 % transfer learning   
          NN = 2; % find the neigbhours, NN is the number of neigbhours
          Nbs = findneighbours(CMA,POP,iter,maxIter,BI,NN);
          LCMA = TransLearning(CMA,POP,LCMA,Nbs,BI);      
     end
  end

%% transfer learning for lower level CMA-ES
function LCMA = TransLearning(CMA,POP,LCMA,Nbs,BI)   
for i = 1 : CMA.lambda
    TCMA(i).C  = LCMA(i).C;  
    TCMA(i).xmean  = LCMA(i).xmean;
end
 for i = 1 : CMA.lambda      
     if POP(i).RF ~= true&&LCMA(i).TR == 1
        Temp_C = zeros(BI.l_dim,BI.l_dim);
        Temp_xm = zeros(1,BI.l_dim);
        value = [Nbs(i).value]/sum([Nbs(i).value]);
        NN  = size(value,2);
        T = 1;
        while (1/(((1 + tan(pi/2*value(NN)))^T))>0.1)
            T = T+0.1;
        end
        index = [Nbs(i).ind];
        for k = 1 : NN
            if  k==1||POP(index(k)).RF == true
                NLR(k) = 1/(((1 + tan(pi/2*value(k)))^T)); 
            else
                NLR(k) = 0;
            end
        end      
    
         Nor_LR = NLR/sum(NLR); 
         for j = 1 : NN     
            Temp_C =  Temp_C + Nor_LR(j)*TCMA(index(j)).C;
            Temp_xm = Temp_xm + Nor_LR(j)*TCMA(index(j)).xmean;
         end
      % feature transfer
       LCMA(i).C  =  Temp_C;   
       LCMA(i).xmean =  Temp_xm;     
       LCMA(i).C = triu(LCMA(i).C) + triu(LCMA(i).C,1)';     
       [LCMA(i).B,LCMA(i).D] = eig(LCMA(i).C);
       LCMA(i).D = sqrt(diag(LCMA(i).D))';
       LCMA(i) = repairCMA(LCMA(i));
       LCMA(i).invsqrtC = LCMA(i).B * diag(LCMA(i).D.^-1) * LCMA(i).B'; 
     end  
 end  
%%
function cfit_ = combineConstraintWithFitness(fit_,c)
if all(c==0)
	cfit_ = fit_;
else
    isFeasible = c == 0;
    if any(isFeasible)
        cfit_(isFeasible) = fit_(isFeasible);
        cfit_(~isFeasible) = min(fit_(isFeasible)) - c(~isFeasible);
    elseif mean(abs(fit_))>50*mean(c)||length(fit_)==2
         cfit_ = - c;
    else
         fit_ = fit_./max([1,abs(fit_)]);
         cfit_ = fit_ - c;
    end 
end

function POP = assignLowerFitness(POP)
fit_ = combineConstraintWithFitness([POP.LF],[POP.LC]);
for i = 1 : length(POP)
    POP(i).fit = fit_(i);
end

function POP = assignUpperFitness(POP,BI)
CV = [POP.UC];
if ~BI.isLowerLevelConstraintsIncludedInUpperLevel
    CV = CV + [POP.LC];
end
% CV = CV / max(CV);
fit_ = combineConstraintWithFitness([POP.UF],CV);
for i = 1 : length(POP)
	POP(i).fit = fit_(i);
	% ????? how to handle RF
	% according to latest experimental results, incorporating RF here seems to be meaningless
end

function Q = refine(P,CMA,BI)
Q = P;
[Q.LX,Q.LF,Q.LC,Q.RF] = RefineSearch(Q.UX,CMA,BI);
if lowerLevelComparator(Q,P)
    Q.RF = max(Q.RF,P.RF);
    [Q.UF,Q.UC] = UL_evaluate(Q.UX,Q.LX,BI.fn);
else
	Q = P;
end

function isNoWorseThan = upperLevelComparator(P,Q,BI)
if isempty(Q)
    isNoWorseThan = true;
else
    tmp = assignUpperFitness([P Q],BI);
    isNoWorseThan = tmp(1).fit >= tmp(2).fit;
end

function isNoWorseThan = lowerLevelComparator(P,Q)
if isempty(Q)
    isNoWorseThan = true;
else
    tmp = assignLowerFitness([P Q]);
    isNoWorseThan = tmp(1).fit >= tmp(2).fit;
end

function LCMA = initLCMAES(CMA,BI,xmean)
% LCMA.xmean = xmean;
LCMA.xmean = CMA.xmean(BI.u_dim+1:end);
LCMA.sigma = 1;
% if CMA.sigma>0.1    
   LCMA.C = CMA.C(BI.u_dim+1:end,BI.u_dim+1:end) * CMA.sigma^2;
% else
%    LCMA.C = CMA.C(BI.u_dim+1:end,BI.u_dim+1:end);
% end
% if all(all(abs(LCMA.C)<1e-6))
%     LCMA.C = LCMA.C + diag(ones(1,BI.l_dim));
% end
LCMA.pc = CMA.pc(BI.u_dim+1:end) * CMA.sigma;
LCMA.ps = zeros(1,BI.l_dim);
LCMA.lambda = 4+floor(3*log(BI.l_dim));
LCMA.mu = floor(LCMA.lambda/2);
weights = log(LCMA.mu+1/2)-log(1:LCMA.mu);
LCMA.weights = weights/sum(weights);
LCMA.mueff=sum(LCMA.weights)^2/sum(LCMA.weights.^2);
LCMA.cc = (4+LCMA.mueff/BI.l_dim) / (BI.l_dim+4 + 2*LCMA.mueff/BI.l_dim);
LCMA.cs = (LCMA.mueff+2) / (BI.l_dim+LCMA.mueff+5);
LCMA.c1 = 2 / ((BI.l_dim+1.3)^2+LCMA.mueff);
LCMA.cmu = min(1-LCMA.c1, 2 * (LCMA.mueff-2+1/LCMA.mueff) / ((BI.l_dim+2)^2+LCMA.mueff));
LCMA.damps = 1 + 2*max(0, sqrt((LCMA.mueff-1)/(BI.l_dim+1))-1) + LCMA.cs;
LCMA.chiN=BI.l_dim^0.5*(1-1/(4*BI.l_dim)+1/(21*BI.l_dim^2));
[LCMA.B,LCMA.D] = eig(LCMA.C);
LCMA.D = sqrt(diag(LCMA.D))';
LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';
LCMA.cy = sqrt(BI.l_dim)+2*BI.l_dim/(BI. l_dim+2);
LCMA.bestIndv = [];      
LCMA.TR  = 0; % learning rate
LCMA.LR  = 0; % learning rate

function LCMA = OnestepLS(xu,LCMA,BI)
%    [LCMA.B,LCMA.D] = eig(LCMA.C);
%    LCMA.D = sqrt(diag(LCMA.D))';
%    LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';
%    cy = sqrt(BI.l_dim)+2*BI.l_dim/(BI.l_dim+2);
for i = 1 : LCMA.lambda
	    Q(i).LX = LCMA.xmean + LCMA.sigma * randn(1,BI.l_dim) .* LCMA.D * LCMA.B';
	    flag = Q(i).LX > BI.l_ub;
	    Q(i).LX(flag) = (LCMA.xmean(flag) + BI.l_ub(flag))/2;
	    flag = Q(i).LX < BI.l_lb;
	    Q(i).LX(flag) = (LCMA.xmean(flag) + BI.l_lb(flag))/2;
	    [Q(i).LF,Q(i).LC] = LL_evaluate(xu,Q(i).LX,BI.fn);
end
    Q = assignLowerFitness(Q);
	[~, Xindex] = sort(-[Q.fit]);
	xold = LCMA.xmean;
	X = cell2mat(arrayfun(@(q)q.LX,Q','UniformOutput',false));
	Y = bsxfun(@minus,X(Xindex(1:LCMA.mu),:),xold) / LCMA.sigma;
	Y = bsxfun(@times,Y,min(1,LCMA.cy./sqrt(sum((Y*LCMA.invsqrtC').^2,2))));
	delta_xmean = LCMA.weights * Y;
	LCMA.xmean = LCMA.xmean + delta_xmean * LCMA.sigma;
	C_mu = Y' * diag(LCMA.weights) * Y;
	LCMA.ps = (1-LCMA.cs)*LCMA.ps + sqrt(LCMA.cs*(2-LCMA.cs)*LCMA.mueff) * delta_xmean * LCMA.invsqrtC;
	LCMA.pc = (1-LCMA.cc)*LCMA.pc + sqrt(LCMA.cc*(2-LCMA.cc)*LCMA.mueff) * delta_xmean;
	LCMA.C = (1-LCMA.c1-LCMA.cmu) * LCMA.C + LCMA.c1 * (LCMA.pc'*LCMA.pc) + LCMA.cmu * C_mu; 
	delta_sigma = (LCMA.cs/LCMA.damps)*(norm(LCMA.ps)/LCMA.chiN - 1);
	LCMA.sigma = LCMA.sigma * exp(min(1,delta_sigma));
	LCMA.C = triu(LCMA.C) + triu(LCMA.C,1)';
	[LCMA.B,LCMA.D] = eig(LCMA.C);
	LCMA.D = sqrt(diag(LCMA.D))';
    LCMA = repairCMA(LCMA);
	LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';   
    if lowerLevelComparator(Q(Xindex(1)), LCMA.bestIndv)       
        LCMA.bestIndv = Q(Xindex(1));
    end



function [bestLX,bestLF,bestLC,bestRF] = RefineSearch(xu,CMA,BI)
sigma0 = 1;
LCMA.xmean = CMA.xmean(BI.u_dim+1:end);
LCMA.sigma = sigma0;
LCMA.C = CMA.C(BI.u_dim+1:end,BI.u_dim+1:end) * CMA.sigma^2;
LCMA.pc = CMA.pc(BI.u_dim+1:end) * CMA.sigma;
LCMA.ps = zeros(1,BI.l_dim);
lambda = 4+floor(3*log(BI.l_dim));
mu = floor(lambda/2);
weights = log(mu+1/2)-log(1:mu);
weights = weights/sum(weights);
mueff=sum(weights)^2/sum(weights.^2);
cc = (4+mueff/BI.l_dim) / (BI.l_dim+4 + 2*mueff/BI.l_dim);
cs = (mueff+2) / (BI.l_dim+mueff+5);
c1 = 2 / ((BI.l_dim+1.3)^2+mueff);
cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((BI.l_dim+2)^2+mueff));
damps = 1 + 2*max(0, sqrt((mueff-1)/(BI.l_dim+1))-1) + cs;
chiN=BI.l_dim^0.5*(1-1/(4*BI.l_dim)+1/(21*BI.l_dim^2));
[LCMA.B,LCMA.D] = eig(LCMA.C);
LCMA.D = sqrt(diag(LCMA.D))';
LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';
cy = sqrt(BI.l_dim)+2*BI.l_dim/(BI.l_dim+2);
bestIndv = [];
bestRF = false;

maxIter = ceil(BI.LmaxFEs / lambda);
ImprIter = ceil(BI.LmaxImprFEs / lambda);
record = zeros(1,maxIter);
for iter = 1 : maxIter
	for i = 1 : lambda
	    Q(i).LX = LCMA.xmean + LCMA.sigma * randn(1,BI.l_dim) .* LCMA.D * LCMA.B';
	    flag = Q(i).LX > BI.l_ub;
	    Q(i).LX(flag) = (LCMA.xmean(flag) + BI.l_ub(flag))/2;
	    flag = Q(i).LX < BI.l_lb;
	    Q(i).LX(flag) = (LCMA.xmean(flag) + BI.l_lb(flag))/2;
	    [Q(i).LF,Q(i).LC] = LL_evaluate(xu,Q(i).LX,BI.fn);
	end
	Q = assignLowerFitness(Q);
	[~, Xindex] = sort(-[Q.fit]);
	xold = LCMA.xmean;
	X = cell2mat(arrayfun(@(q)q.LX,Q','UniformOutput',false));
	Y = bsxfun(@minus,X(Xindex(1:mu),:),xold) / LCMA.sigma;
	Y = bsxfun(@times,Y,min(1,cy./sqrt(sum((Y*LCMA.invsqrtC').^2,2))));
	delta_xmean = weights * Y;
	LCMA.xmean = LCMA.xmean + delta_xmean * LCMA.sigma;
	C_mu = Y' * diag(weights) * Y;
	LCMA.ps = (1-cs)*LCMA.ps + sqrt(cs*(2-cs)*mueff) * delta_xmean * LCMA.invsqrtC;
	LCMA.pc = (1-cc)*LCMA.pc + sqrt(cc*(2-cc)*mueff) * delta_xmean;
	LCMA.C = (1-c1-cmu) * LCMA.C + c1 * (LCMA.pc'*LCMA.pc) + cmu * C_mu; 
	delta_sigma = (cs/damps)*(norm(LCMA.ps)/chiN - 1);
	LCMA.sigma = LCMA.sigma * exp(min(CMA.delta_sigma_max,delta_sigma));
	LCMA.C = triu(LCMA.C) + triu(LCMA.C,1)';
	[LCMA.B,LCMA.D] = eig(LCMA.C);
	LCMA.D = sqrt(diag(LCMA.D))';
    LCMA = repairCMA(LCMA);
	LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';

	% elite-preservation right????
	if lowerLevelComparator(Q(Xindex(1)),bestIndv)
		bestIndv = Q(Xindex(1));
	end
    record(iter) = bestIndv.LF;

    if (iter>ImprIter && abs(record(iter)-record(iter-ImprIter+1))/(abs(record(1))+abs(record(iter))) < 1e-4) ...
		|| (iter > ImprIter && abs(record(iter) - record(iter-ImprIter+1)) < 10*BI.l_ftol) ...
        || LCMA.sigma / sigma0 < 1e-2 ...
        || LCMA.sigma / sigma0 > 1e2
    	bestRF = true;
        break; 
    end
end
bestLX = bestIndv.LX;
bestLF = bestIndv.LF;
bestLC = bestIndv.LC;


function CMA = initCMAES(BI)
CMA.lambda = 4+floor(3*log(BI.dim));
% CMA.sigma = min(BI.xrange(2,:)-BI.xrange(1,:))/2;
CMA.sigma = 0.3*median(BI.xrange(2,:)-BI.xrange(1,:));
CMA.mu = floor(CMA.lambda/2);
CMA.weights = log(CMA.mu+1/2)-log(1:CMA.mu);
CMA.weights = CMA.weights/sum(CMA.weights);
CMA.mueff=sum(CMA.weights)^2/sum(CMA.weights.^2);
CMA.cc = (4+CMA.mueff/BI.dim) / (BI.dim+4 + 2*CMA.mueff/BI.dim);
CMA.cs = (CMA.mueff+2) / (BI.dim+CMA.mueff+5);
CMA.c1 = 2 / ((BI.dim+1.3)^2+CMA.mueff);
CMA.cmu = min(1-CMA.c1, 2 * (CMA.mueff-2+1/CMA.mueff) / ((BI.dim+2)^2+CMA.mueff));
CMA.damps = 1 + 2*max(0, sqrt((CMA.mueff-1)/(BI.dim+1))-1) + CMA.cs;
CMA.chiN=BI.dim^0.5*(1-1/(4*BI.dim)+1/(21*BI.dim^2));
CMA.pc = zeros(1,BI.dim);
CMA.ps = zeros(1,BI.dim);
CMA.B = eye(BI.dim);
CMA.D = ones(1,BI.dim);
CMA.C = CMA.B * diag(CMA.D.^2) * CMA.B';
CMA.invsqrtC = CMA.B * diag(CMA.D.^-1) * CMA.B';
CMA.xmean = (BI.xrange(2,:)-BI.xrange(1,:)).*rand(1,BI.dim)+BI.xrange(1,:);
CMA.cy = sqrt(BI.dim)+2*BI.dim/(BI.dim+2);
CMA.delta_sigma_max = 1;

function CMA = updateCMAES(CMA,POP,BI)
[~, Xindex] = sort(-[POP.fit]);
xold = CMA.xmean;
X = cell2mat(arrayfun(@(p)[p.UX p.LX],POP(Xindex(1:CMA.mu))','UniformOutput',false));
Y = bsxfun(@minus,X,xold)/CMA.sigma;
Y = bsxfun(@times,Y,min(1,CMA.cy./sqrt(sum((Y*CMA.invsqrtC').^2,2))));
delta_xmean = CMA.weights * Y;
CMA.xmean = CMA.xmean + delta_xmean * CMA.sigma;
C_mu = Y' * diag(CMA.weights) * Y;
CMA.ps = (1-CMA.cs)*CMA.ps + sqrt(CMA.cs*(2-CMA.cs)*CMA.mueff) * delta_xmean * CMA.invsqrtC;
CMA.pc = (1-CMA.cc)*CMA.pc + sqrt(CMA.cc*(2-CMA.cc)*CMA.mueff) * delta_xmean;
CMA.C = (1-CMA.c1-CMA.cmu) * CMA.C + CMA.c1 * (CMA.pc'*CMA.pc) + CMA.cmu * C_mu; 
delta_sigma = (CMA.cs/CMA.damps)*(norm(CMA.ps)/CMA.chiN - 1);
CMA.sigma = CMA.sigma * exp(min(delta_sigma,CMA.delta_sigma_max));
CMA.C = triu(CMA.C) + triu(CMA.C,1)';
[CMA.B,CMA.D] = eig(CMA.C);
CMA.D = sqrt(diag(CMA.D))';
CMA = repairCMA(CMA);
CMA.invsqrtC = CMA.B * diag(CMA.D.^-1) * CMA.B';

function [F,C] = LL_evaluate(xu,xl,fn)
[F,~,C] = llTestProblem(xl,fn,xu);
C = sum(max(0,C));

function [F,C] = UL_evaluate(UPop,LPOP,fn)
[F,~,C] = ulTestProblem(UPop, LPOP, fn);
C = sum(max(0,C));

function Model = repairCMA(Model)
dim = length(Model.D);
% limit condition of C to 1e14
if any(Model.D<=0)
    Model.D(Model.D<0) = 0;
    tmp = max(Model.D)/1e7;
    Model.C = Model.C + tmp * eye(dim);
    Model.D = Model.D + tmp * ones(1,dim);
end
if max(Model.D) > 1e7 * min(Model.D)
    tmp = max(Model.D)/1e7 - min(Model.D);
    Model.C = Model.C + tmp * eye(dim);
    Model.D = Model.D + tmp * ones(1,dim);
end
% rescale sigma
if Model.sigma > 1e7 * max(Model.D)
    fac = Model.sigma / max(Model.D);
    Model.sigma = Model.sigma / fac;
    Model.D = Model.D * fac;
    Model.pc = Model.pc * fac;
    Model.C = Model.C * fac^2;
end