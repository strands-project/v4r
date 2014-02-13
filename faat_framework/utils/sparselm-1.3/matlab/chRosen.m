function x = chRosen(p, n, nnz)
% nnz is unused

  for k=1:n
    i=(fix((k+1)/2));
    if mod(k, 2)==1
      x(k)=10.0*(p(i)*p(i)-p(i+1));
    else
      x(k)=p(i)-1.0;
    end
  end
