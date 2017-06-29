function test_suite = test_get_scene_counts_as_perc
  initTestSuite;
end

function test_with_canonical_filenames
  scenes = cell(10, 1);
  scenes{1} = 'bathroom_0001';
  scenes{2} = 'bathroom_0002';
  scenes{3} = 'bedroom_0002';
  scenes{4} = 'bedroom_0003';
  scenes{5} = 'living_room_0004';
  scenes{6} = 'bathroom_0001';
  scenes{7} = 'bathroom_0001';
  scenes{8} = 'bedroom_0002';
  scenes{9} = 'living_room_0004';
  scenes{10} = 'living_room_0004';
  [sceneCounts, sceneTypeCounts] = get_scene_counts(scenes);
  sceneCounts = get_scene_counts_as_perc(sceneCounts, sceneTypeCounts);
  
  assertEqual(length(sceneCounts), 5);
  assertEqual(sceneCounts('bathroom_0001'), 3/4);
  assertEqual(sceneCounts('bathroom_0002'), 1/4);
  assertEqual(sceneCounts('bedroom_0002'), 2/3);
  assertEqual(sceneCounts('bedroom_0003'), 1/3);
  assertEqual(sceneCounts('living_room_0004'), 3/3);
end