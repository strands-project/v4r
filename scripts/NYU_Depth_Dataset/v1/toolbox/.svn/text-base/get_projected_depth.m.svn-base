% Projects the depth values onto the RGB image plane.
%
% Args:
%   depth - the original uint16 output from the Kinect. The bytes MUST have
%           already been swapped via swapbytes.m
%   maxDepth (optional) - the maximum depth from the device. The default
%                         value is 10 meters.
%
% Returns:
%   depthProj - the projected depth image, output in meters (double).
%   depthProjScaled - the depth scaled to fall between 0 and 255 (uint8).
function [depthProj, depthProjScaled] = get_projected_depth(depth, maxDepth)

  %% Validate the arguments.
  depth = double(depth);  

  if ~exist('maxDepth', 'var') || isempty(maxDepth)
    maxDepth = 10;
  else
    assert(isscalar(maxDepth), 'maxDepth must be a scalar value (in meters)');
  end

  [H, W, ~] = size(depth);
  [xx, yy] = meshgrid(1:W,1:H);

  %% Color camera parameters
  fx_rgb = 5.1930334103339817e+02;
  fy_rgb = 5.1816401430246583e+02;
  cx_rgb = 3.2850951551345941e+02;
  cy_rgb = 2.5282555217253503e+02;
 
  %% Depth camera parameters
  fx_d =  5.7616540758591043e+02;
  fy_d = 5.7375619782082447e+02;
  cx_d = 3.2442516903961865e+02;
  cy_d = 2.3584766381177013e+02;
 
  % Rotation matrix
  R =  inv([  9.9998579449446667e-01, 3.4203777687649762e-03, -4.0880099301915437e-03;
             -3.4291385577729263e-03, 9.9999183503355726e-01, -2.1379604698021303e-03;
              4.0806639192662465e-03, 2.1519484514690057e-03,  9.9998935859330040e-01]);
   
  % Translation vector.         
  T = -[  2.2142187053089738e-02, -1.4391632009665779e-04, -7.9356552371601212e-03 ]';
            
  fc_d = [fx_d,fy_d];
  cc_d = [cx_d,cy_d];
  
  fc_rgb = [fx_rgb,fy_rgb];
  cc_rgb = [cx_rgb,cy_rgb];

  % 1. raw depth --> absolute depth in meters.
  depth2 = 0.3513e3./(1.0925e3-depth); %%% nathan's data
  
  depth2(depth2>maxDepth) = maxDepth;
  depth2(depth2<0) = 0;
    
  % 2. points in depth image to 3D world points:
  x3 = (xx - cc_d(1)) .* depth2 / fc_d(1);
  y3 = (yy - cc_d(2)) .* depth2 / fc_d(2);
  z3 = depth2;
  
  % 3. now rotate & translate the 3D points
  p3 = [x3(:),y3(:),z3(:)]';
  p3_new = R*p3 + T*ones(1,length(p3));

  x3_new = reshape(p3_new(1,:)',[H,W]);
  y3_new = reshape(p3_new(2,:)',[H,W]);
  z3_new = reshape(p3_new(3,:)',[H,W]);
  
  % 4. project into rgb coordinate frame
  x_proj = (x3_new .* fc_rgb(1) ./ z3_new) + cc_rgb(1);
  y_proj = (y3_new .* fc_rgb(2) ./ z3_new) + cc_rgb(2);
 
  %% now project back to actual image
  
  x_proj = round(x_proj);
  y_proj = round(y_proj);
  
  g_ind = find(x_proj(:)>0 & x_proj(:)<size(depth,2) & y_proj(:)>0 & y_proj(:)<size(depth,1));
  
  depthProj = zeros(size(depth));
  [depth_sorted,order] = sort(-depth2(g_ind));
  depth_sorted = - depth_sorted;
  
  % z-buffer projection
  for i=1:length(order)
    depthProj(y_proj(g_ind(order(i))),x_proj(g_ind(order(i)))) = depth_sorted(i);
  end
  
  %% Fix weird values...
  q = depthProj > maxDepth;
  depthProj(q) = maxDepth;
  
  q = depthProj < 0;
  depthProj(q) = 0;
  
  q = isnan(depthProj);
  depthProj(q) = 0;

  %% Rescale to uint8
  depthProjScaled = uint8(depthProj * 255 / maxDepth);
end
