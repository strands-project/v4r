% Counts the number of images per scene and scene type.
% 
% Args:
%   scens - cell array of 
%
% Returns:
%   sceneCounts - a map from scene to the number of images in that scene.
%   sceneTypeCounts - a map from sceneType to the number of images in that
%                     sceneType.
function [sceneCounts, sceneTypeCounts] = get_scene_counts(scenes)
  sceneCounts = containers.Map();
  sceneTypeCounts = containers.Map();

  for ii = 1 : numel(scenes)
    scene = scenes{ii};
    sceneType = get_scene_type_from_scene(scene);
    
    if ~isKey(sceneCounts, scene)
      sceneCounts(scene) = 0;
    end
    sceneCounts(scene) = sceneCounts(scene) + 1;
    
    if ~isKey(sceneTypeCounts, sceneType)
      sceneTypeCounts(sceneType) = 0;
    end
    sceneTypeCounts(sceneType) = sceneTypeCounts(sceneType) + 1;
  end
end