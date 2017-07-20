% Calculates the percentage (in terms of number of images) that each scene
% makes up out of its scene type.
%
% Args:
%   sceneCounts - map from scene to image files per scene
%   sceneTypeCounts - map from sceneType to image files per sceneType
%
% Returns:
%   sceneCounts - map from scene to percentage of sceneType the scene makes
%                 up (in terms of total image files).
function sceneCounts = get_scene_counts_as_perc(sceneCounts, sceneTypeCounts)
  scenes = keys(sceneCounts);
  for ii = 1 : length(sceneCounts)
    scene = scenes{ii};
    sceneType = get_scene_type_from_scene(scene);
    sceneCounts(scene) = sceneCounts(scene) / sceneTypeCounts(sceneType);
  end
end