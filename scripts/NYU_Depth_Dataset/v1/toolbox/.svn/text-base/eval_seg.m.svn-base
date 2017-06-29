% Evaluates the accuracy of a given segmentation using pixel-level
% accuracy.
%
% Args:
%   predictions - a HxW matrix of labels, where H and W are the height and
%                 width of the image, respectively.
%   groundTruth - a HxW matrix of labels, where H and W are the height and
%                 width of the image, respectively.
%   numClasses - the number of total classes in the experiment.
function [accuracy, numCorrect, numLabeled, cm] = eval_seg(predictions, ...
    groundTruth, numClasses)
  
  numCorrect = nnz(predictions == groundTruth);
  numLabeled = nnz(groundTruth > 0);
  
  accuracy = numCorrect / numLabeled;
  
  cm = confusion_matrix(groundTruth(:), predictions(:), numClasses);
end