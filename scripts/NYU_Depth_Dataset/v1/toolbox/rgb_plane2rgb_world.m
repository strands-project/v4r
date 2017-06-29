function points3d = rgb_plane2rgb_world(imgDepth)
  % Color camera parameters
  fx_rgb = 5.1930334103339817e+02;
  fy_rgb = 5.1816401430246583e+02;
  cx_rgb = 3.2850951551345941e+02;
  cy_rgb = 2.5282555217253503e+02;

  [H, W] = size(imgDepth);

  % Make the original consistent with the camera location:
  [xx, yy] = meshgrid(1:W, 1:H);

  x3 = (xx - cx_rgb) .* imgDepth / fx_rgb;
  y3 = (yy - cy_rgb) .* imgDepth / fy_rgb;
  z3 = imgDepth;
  
  points3d = [x3(:) y3(:) z3(:)];
end
