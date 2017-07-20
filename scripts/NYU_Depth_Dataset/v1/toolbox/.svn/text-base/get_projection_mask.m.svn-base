% Gets a mask for the projected images that is most conservative with
% regard the regions that maintain the original kinect signal.
function mask = get_projection_mask()
  mask = false(480, 640);
  mask(45:470, 36:600) = 1;
end