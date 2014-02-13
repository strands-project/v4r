% demonstrates two ways of supplying a sparse Jacobian,
% i.e. directly and through (i, j, val) triplets

function jac = jacchRosen(p, n, nnz)
  jac=jacchRosen_dir(p, n, nnz);
  %jac=jacchRosen_tri(p, n, nnz);


% the Jacobian is directly created
function jac = jacchRosen_dir(p, n, nnz)
  m=max(size(p));

  % create an empty sparse matrix
  jac=sparse([], [], [], n, m, nnz);
  for k=1:n
    i=(fix((k+1)/2));
    % fill in nonzero elements
    if mod(k, 2)==1
      jac(k, i)=20.0*p(i);
      jac(k, i+1)=-10.0;
    else
      jac(k, i)=1.0;
    end
  end


% alternatively, the Jacobian is first created in sparse
% triplet format and then converted to a matlab sparse matrix (CCS)
function jac = jacchRosen_tri(p, n, nnz)
  m=max(size(p));

  % preallocate triplet vectors
  rowidx=zeros(nnz, 1);
  colidx=zeros(nnz, 1); 
  val=zeros(nnz, 1); 
  l=1;
  for k=1:n
    i=(fix((k+1)/2));
    if mod(k, 2)==1
      val(l)=20.0*p(i); rowidx(l)=k; colidx(l)=i; l=l+1;
      val(l)=-10.0;     rowidx(l)=k; colidx(l)=i+1; l=l+1;
    else
      val(l)=1.0;  rowidx(l)=k; colidx(l)=i; l=l+1;
    end
  end

  % convert to sparse matrix
  jac=sparse(rowidx, colidx, val, n, m, l);

