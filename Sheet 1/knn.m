function [ nbs ] = knn( spc,pnt,k, norm_p )
%KNN Summary of this function goes here
%   Detailed explanation goes here
  nbs = [];
  nonb = 0;
  while nonb < k
    if nonb == 0
      candidates = spc;
    else
      candidates = transpose(setdiff(transpose(spc),transpose(nbs),'rows'));
    end
    closestp = [0,0,0];
    closestn = inf;
    for i = 1:length(candidates)
      d =transpose(pnt-candidates(:,i));
      n=norm(d,norm_p);
      if n < closestn
        closestn = n;
        closestp = candidates(:,i);
      end
    end
    nonb = nonb+1;
    nbs(:,nonb) = closestp;
  end
end

