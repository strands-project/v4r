% Demo for 'get_train_test_split' which shows that images of the same scene
% are never split across both train and test.

PATH_TO_DATASET = '~/data/kinect/datasets/'; % Fill in here.

load(sprintf('%s/nyu_depth_data_labeled_2011_05_31.mat', PATH_TO_DATASET), 'scenes');

[trainNdxs, testNdxs, trainScenes, testScenes] = get_train_test_split(scenes);

fprintf('Training set size: %d.\n', numel(trainNdxs));
fprintf('Testing set size: %d.\n', numel(testNdxs));
fprintf('\n');

fprintf('Training Scenes: \n');
for ii = 1 : numel(trainScenes)
  fprintf(sprintf('%s\n', trainScenes{ii}));
end

fprintf('\n');
fprintf('Testing Scenes: \n');
for ii = 1 : numel(testScenes)
  fprintf(sprintf('%s\n', testScenes{ii}));
end