% This is the main framework of TLEA including TLEA-CMA-ES and TLEA-DE.
% Copyright (c) 2021, Lei Chen@GDUT CISlab. All rights reserved.
% We would like to thank Dr. Xiaoyu He from Sun Yat-Sen University for
% offering the source code of BL-CMA-ES.
% Please note this is a pre-alpha version. 
% If you have any questions, please don not hesitate to ask me through
% email:chenaction@126.com, or
% Wechat:fengluowusheng2008
%% test instance
addpath(genpath(pwd));
clear; close; clc;
maxRuns = 19;
name_func  = {'SMD','TP'};
num_func = [12,10];
dim_func = 20; %for SMD instances
for seq =1:1
    name_f  = char(name_func(seq));
    num_f   = num_func(seq);
    if seq ==1
       BI_list = getBLOPinfo(name_f,1:num_f,dim_func);
    else 
       BI_list = getBLOPinfo(name_f,1:num_f);
    end
    for BI = BI_list'
        BI= getparams(BI);
        for runNo = 1:maxRuns
            tic;
%             ins = TLEACMAES(BI);
            ins = TLEADE(BI);
            ins.runNo = runNo;
            ins.runTime = toc;
            fprintf('%s %s #%d [%g,%g]\n', ins.alg, ins.BI.fn, ins.runNo, ins.UF, ins.LF);
        end
    end
end