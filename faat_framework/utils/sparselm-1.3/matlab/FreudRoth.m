function x = FreudRoth(p, n, nnz)
% nnz is unused

  for k=1:n
    i=fix((k+1)/2);
    if mod(k, 2)==1
      x(k)=p(i)+p(i+1)*((5.0-p(i+1))*p(i+1)-2.0) - 13.0;
    else
      x(k)=p(i)+p(i+1)*((1.0+p(i+1))*p(i+1)-14.0) - 29.0;
    end
  end
