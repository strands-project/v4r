function test_suite = test_get_scene_type_from_scene
  initTestSuite;
end

function test_with_simple_scene_name
  sceneType = get_scene_type_from_scene('hallway_0001');
  assertEqual(sceneType, 'hallway');
end

function test_with_complex_scene_name
  sceneType = get_scene_type_from_scene('dept_store_0002');
  assertEqual(sceneType, 'dept_store');
end

function test_with_complex_scene_name2
  sceneType = get_scene_type_from_scene('living_room_0001b');
  assertEqual(sceneType, 'living_room');
end