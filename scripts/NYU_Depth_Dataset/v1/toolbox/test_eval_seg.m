function test_suite = test_eval_seg()
  initTestSuite;
end

function test_all_correct()
  numClasses = 5;
  groundTruth = randi(numClasses, [3 3]);
  predictions = groundTruth;
  
  [accuracy, numCorrect, numLabeled] = eval_seg(predictions, ...
      groundTruth, numClasses);
  assertEqual(accuracy, 1.0);
  assertEqual(numCorrect, 9);
  assertEqual(numLabeled, 9);
end

function test_all_incorrect()
  numClasses = 5;
  groundTruth = randi(numClasses, [3 3]);
  predictions = groundTruth + 1;
  
  [accuracy, numCorrect, numLabeled] = eval_seg(predictions, ...
      groundTruth, numClasses);
  assertEqual(accuracy, 0.0);
  assertEqual(numCorrect, 0);
  assertEqual(numLabeled, 9);
end

function test_some_correct()
  numClasses = 5;
  groundTruth = randi(numClasses, [3 3]);
  incorrect = logical(randi(2, [3 3]) - 1);
  predictions = groundTruth + incorrect;
  
  [accuracy, numCorrect, numLabeled] = eval_seg(predictions, ...
      groundTruth, numClasses);
  assertEqual(accuracy, nnz(~incorrect) / 9);
  assertEqual(numCorrect, 9-nnz(incorrect));
  assertEqual(numLabeled, 9);
end

function test_some_correct_some_missing()
  numClasses = 3;
  groundTruth = [0 1 1;
                 2 0 1;
                 0 3 2];
  predictions = [2 1 2;
                 2 3 1;
                 2 3 3];
  [accuracy, numCorrect, numLabeled] = eval_seg(predictions, ...
      groundTruth, numClasses);
  assertEqual(accuracy, 4/6);
  assertEqual(numCorrect, 4);
  assertEqual(numLabeled, 6);
end