% demonstrates two ways of supplying a sparse Jacobian,
% i.e. directly and through (i, j, val) triplets

function jac = jacchWood_zp(p, n, nnz)
  jac=jacchWood_dir(p, n, nnz);
  %jac=jacchWood_tri(p, n, nnz);


% the Jacobian is directly created
function jac = jacchWood_dir(p, n, nnz)
  m=max(size(p));

  % create an empty sparse matrix
  jac=sparse([], [], [], n, m, nnz);
  for k=1:n
    i=2*(fix((k+5)/6))-1;
    % actual values below are unimportant, any nonzero values would suffice
    switch mod(k, 6)
      case 0
        jac(k, i+1)=1.0;
        jac(k, i+3)=1.0;
      case 1
        jac(k, i)=1.0;
        jac(k, i+1)=1.0;
      case 2
        jac(k, i)=1.0;
      case 3
        jac(k, i+2)=1.0;
        jac(k, i+3)=1.0;
      case 4
        jac(k, i+2)=1.0;
      case 5
        jac(k, i+1)=1.0;
        jac(k, i+3)=1.0;
    end
  end


% alternatively, the Jacobian is first created in sparse
% triplet format and then converted to a matlab sparse matrix (CCS)
function jac = jacchWood_tri(p, n, nnz)
  m=max(size(p));

  % preallocate triplet vectors
  rowidx=zeros(nnz, 1);
  colidx=zeros(nnz, 1);
  l=1;
  for k=1:n
    i=2*(fix((k+5)/6))-1;
    % only the row and column indices of nonzero Jacobian elements are specified below
    switch mod(k, 6)
      case 0
        rowidx(l)=k; colidx(l)=i+1; l=l+1;
        rowidx(l)=k; colidx(l)=i+3; l=l+1;
      case 1
        rowidx(l)=k; colidx(l)=i; l=l+1;
        rowidx(l)=k; colidx(l)=i+1; l=l+1;
      case 2
        rowidx(l)=k; colidx(l)=i; l=l+1;
      case 3
        rowidx(l)=k; colidx(l)=i+2; l=l+1;
        rowidx(l)=k; colidx(l)=i+3; l=l+1;
      case 4
        rowidx(l)=k; colidx(l)=i+2; l=l+1;
      case 5
        rowidx(l)=k; colidx(l)=i+1; l=l+1;
        rowidx(l)=k; colidx(l)=i+3; l=l+1;
    end
  end

  % convert to sparse matrix
  % any nonzero numbers would suffice for the values vectors below; use a vector of 1's
  jac=sparse(rowidx, colidx, ones(l-1, 1), n, m);
