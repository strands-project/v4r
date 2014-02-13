% Demo program for sparseLM's MEX-file interface

%format long
% Chained Rosenbrock function
% minimum at (1 ... 1)

m=10;
n=2*(m-1);
nnz=3*n/2;
p0=zeros(m, 1); % preallocate
p0(1:2:m)=-1.2; % odd
p0(2:2:m)=1.0; % even

x=zeros(n, 1);

options=[1E-03, 1E-15, 1E-15, 1E-20, 1E-06, sparselm_spsolvr('cholmod')];

disp('Chained Rosenbrock function');
% analytic Jacobian
% n, nnz demonstrate additional data passing
[ret, popt, info]=sparselm('chRosen', 'jacchRosen', 'an', p0, 0, x, nnz, -1, 200, options, n, nnz);
% analytic Jacobian with explicit zero pattern
%S=jacchRosen(p0, n, nnz);
%[ret, popt, info]=sparselm('chRosen', 'jacchRosen', 'anzp', S, p0, 0, x, nnz, -1, 200, options, n, nnz);
fprintf('sparseLM returned %d in %g iter, reason %g, error %g [initial %g], %d/%d func/fjac evals, %d lin. systems\n',...
         ret, info(6), info(7), info(2), info(1), info(8), info(9), info(10));
popt



% Chained Wood function
% minimum at (1 ... 1)

m=18;
n=3*(m-2);
nnz=10*n/6;

p0=zeros(m, 1); % preallocate
for i=1:m
  if(i<=4)
    if(mod(i, 2)==1)
      p0(i)=-3;
    else
      p0(i)=0;
    end
  else      
    if(mod(i, 2)==1)
      p0(i)=-2;
    else
      p0(i)=-1;
    end
  end
end

x=zeros(n, 1);

options=[1E-03, 1E-15, 1E-15, 1E-20, 1E-06, sparselm_spsolvr('cholmod')];

disp('Chained Wood function');
% analytic Jacobian
% n, nnz demonstrate additional data passing
[ret, popt, info]=sparselm('chWood', 'jacchWood_an', 'an', p0, 0, x, nnz, -1, 200, options, n, nnz);
% finite difference Jacobian (only the zero pattern is supplied)
%[ret, popt, info]=sparselm('chWood', 'jacchWood_zp', 'zp', p0, 0, x, nnz, -1, 200, options, n, nnz);
% analytic Jacobian with explicit zero pattern
% S=jacchWood_zp(p0, n, nnz);
% [ret, popt, info]=sparselm('chWood', 'jacchWood_an', 'anzp', S, p0, 0, x, nnz, -1, 200, options, n, nnz);
fprintf('sparseLM returned %d in %g iter, reason %g, error %g [initial %g], %d/%d func/fjac evals, %d lin. systems\n',...
         ret, info(6), info(7), info(2), info(1), info(8), info(9), info(10));
popt


% Extended Freudenstein and Roth function
% minimum at (12.2691, -0.8319, -1.5069, -1.5347, -1.5358, ... , -1.5358 -1.5359, -1.5361, -1.5433)

m=12;
n=2*(m-1);
nnz=4*n/2;
p0=0.5*ones(m, 1);
p0(m)=-2.0;
x=zeros(n, 1);

options=[1E-03, 1E-15, 1E-15, 1E-20, 1E-06, sparselm_spsolvr('cholmod')];

disp('Extended Freudenstein and Roth function');
% analytic Jacobian
% n, nnz demonstrate additional data passing
[ret, popt, info]=sparselm('FreudRoth', 'jacFreudRoth', 'an', p0, 0, x, nnz, -1, 200, options, n, nnz);
fprintf('sparseLM returned %d in %g iter, reason %g, error %g [initial %g], %d/%d func/fjac evals, %d lin. systems\n',...
         ret, info(6), info(7), info(2), info(1), info(8), info(9), info(10));
popt


