% Gets a train-test split from the provided list of scenes.
%
% Args:
%   scenes - Nx1 cell array where N is the number of images in the dataset.
%   trainPerc - (optional) the percentage of the dataset that should be in
%               the training set.
%
% Returns:
%   trainNdxs - Ax1 array of indices into N of training indices.
%   testNdxs - Bx1 array of indices into N of testing indices.
%   trainScenes - Ax1 cell array of the scenes that are a part of the
%                 training set.
%   testScenes - Bx1 cell array of the scenes that are a part of the
%                testing set.
function [trainNdxs, testNdxs, trainScenes, testScenes] = get_train_test_split(scenes, trainPerc)
  error(nargchk(1,2,nargin));
  if nargin < 2
    trainPerc = 0.5;
  end

  % Count the number of scenes and scene types.
  [sceneCounts, sceneTypeCounts] = get_scene_counts(scenes);
  
  % Turn the counts into percentages.
  scenePercs = get_scene_counts_as_perc(sceneCounts, sceneTypeCounts);
  
  [trainScenes, testScenes] = get_split(scenePercs, sceneTypeCounts, trainPerc);
  
  % Just sorts them...
  trainScenes = unique(trainScenes);
  testScenes = unique(testScenes);
  
  trainNdxs = get_inds_of_scenes(scenes, trainScenes);
  testNdxs = get_inds_of_scenes(scenes, testScenes);
end

function [scenesTrain, scenesTest] = get_split(scenePercs, sceneTypeCounts, split)
  
  scenes = keys(scenePercs);
  N = length(scenes);
  
  scenesTrain = cell(N, 1);
  scenesTest = cell(N, 1);
  
  numTrain = 0;
  numTest = 0;

  % The percentage of each scene type that has been assigned to the
  % training set. 
  percOfSceneTypesTrain = containers.Map();
  
  % Initialize this percentage to zero.
  sceneTypes = keys(sceneTypeCounts);
  for ii = 1 : numel(sceneTypes)
    percOfSceneTypesTrain(sceneTypes{ii}) = 0;
  end

  % Randomize the order in which we view the scenes.
  seq = randperm(numel(scenes));
  
  for ii = 1 : numel(scenes)
    scene = scenes{seq(ii)};    
    sceneType = get_scene_type_from_scene(scene);

    curPerc = percOfSceneTypesTrain(sceneType);
    
    % Now determine whether adding the current scene gets us closer or
    % farther from the split goal.
    withoutDiff = abs(curPerc - split);
    withDiff = abs(curPerc + scenePercs(scene) - split);
    
    if withDiff < withoutDiff
      percOfSceneTypesTrain(sceneType) = ...
        percOfSceneTypesTrain(sceneType) + scenePercs(scene);
      numTrain = numTrain + 1;
      scenesTrain{numTrain} = scene;
    else
      numTest = numTest + 1;
      scenesTest{numTest} = scene;
    end
  end

  % Drop any empty cells not used.
  scenesTrain = scenesTrain(1:numTrain);
  scenesTest = scenesTest(1:numTest);
end

function inds = get_inds_of_scenes(allScenes, uniqueScenes)
  N = numel(allScenes);
  inds = zeros(N, 1);
  
  numNdxs = 0;
  
  for nn = 1 : N
    scene = allScenes{nn};
    
    sceneFound = 0;
    for ii = 1 : numel(uniqueScenes)
      if strcmp(scene, uniqueScenes{ii})
        sceneFound = 1;
        break;
      end
    end
    
    if sceneFound
      numNdxs = numNdxs + 1;
      inds(numNdxs) = nn;
    end
  end
  
  inds = inds(1:numNdxs);
end