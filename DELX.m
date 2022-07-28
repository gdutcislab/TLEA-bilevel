function CPOP = DELX(s,BQ,Q,SQ,sigma,cross_rate,LDE)
        CPOP  = Q(s);  
        x= Q(s).LX;      
        r0    = floor(rand*size(BQ,2))+1;
        bestIndv = BQ(r0);
        cent_x = bestIndv.LX;
        d  = length(cent_x);
        r1 = floor(rand*size(Q,2))+1;
        while (r1==s)
            r1=floor(rand*(size(Q,2)))+1;
        end
         r2=floor(rand*(size([Q,SQ],2)))+1;
        while (r2==r1||r2==s)
            r2=floor(rand*(size([Q,SQ],2)))+1;
        end
     if rand < 0.
               [B,D] = eig(LDE.C);
               D = sqrt(diag(D));
        % limit condition of C to 1e20 + 1
          if max(diag(D)) > 1e6*min(diag(D))
             tmp = max(diag(D))/1e6 - min(diag(D));
             LDE.C = LDE.C + tmp*eye(d);
             [B,D] = eig(LDE.C);
          end         
          D = sqrt(diag(D))';
        if r2>size(Q,2)
            v = x - sigma*(x - cent_x).*D*B' + sigma*rand(1,d).*D*B';
        else
            v = x - sigma*(x - cent_x).*D*B' + sigma*rand(1,d).*D*B';
        end
        else
            if r2>size(Q,2)
            v = x - sigma*(x - cent_x) + sigma*(SQ(r2-size(Q,2)).LX-Q(r1).LX);
              else
            v = x - sigma*(x - cent_x) + sigma*(Q(r2).LX-Q(r1).LX);
            end
     end
     %         crossover
        index = [1:d];
        index = index(rand(1,d)<cross_rate);
        J = unidrnd(d);
        cld_x = x;
        cld_x(index) = v(index);
        cld_x(J) = v(J);   
        CPOP.LX =  cld_x;
end