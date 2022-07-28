function BI= getparams(BI)
    if BI.dim == 3
        % for gold mining problem
        BI.UmaxFEs = 1500;
        BI.UmaxImprFEs = 70;
        BI.LmaxFEs = 150;
        BI.LmaxImprFEs = 15;
    elseif BI.dim == 5 || BI.dim == 6 || strcmp(BI.fn(1:2),'tp')
        % for generic benchmark test problem or decision making problem 
        BI.UmaxFEs = 2500;
        BI.UmaxImprFEs = 350;
        BI.LmaxFEs = 250;
        BI.LmaxImprFEs = 25;
    elseif BI.dim == 10
        BI.UmaxFEs = 3500;
        BI.UmaxImprFEs = 500;
        BI.LmaxFEs = 350;
        BI.LmaxImprFEs = 35;
    elseif BI.dim == 20
        BI.UmaxFEs = 5000;
        BI.UmaxImprFEs = 750;
        BI.LmaxFEs = 500;
        BI.LmaxImprFEs = 50;
    else
        error('unknown dimensionality');
    end
    
    BI.u_N = 50;
    BI.l_N = 50;
end