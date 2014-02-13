% the Jacobian is directly created
function jac = jacFreudRoth(p, n, nnz)
  m=max(size(p));

  % create an empty sparse matrix
  jac=sparse([], [], [], n, m, nnz);
  for k=1:n
    i=fix((k+1)/2);
    % fill in nonzero elements
    if mod(k, 2)==1
      jac(k, i)=1.0;
      jac(k, i+1)=(5.0-p(i+1))*p(i+1)-2.0+p(i+1)*(-2.0*p(i+1)+5.0);
    else
      jac(k, i)=1.0;
      jac(k, i+1)=(1.0+p(i+1))*p(i+1)-14.0+p(i+1)*(2.0*p(i+1)+1.0);
    end
  end

