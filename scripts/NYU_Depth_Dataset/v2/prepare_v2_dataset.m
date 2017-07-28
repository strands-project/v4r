% This script converts the labeled RGBD frames of NYUv2 into pcd pointcloud
% files readable by PCL.
% Note: groundtruth labels are already mapped with labeling from
% Couprie et al.: C. Couprie, C. Farabet, L. Najman, and Y. LeCun.
% Indoor semantic segmentation using depth information. ICLR 2013
% ATTENTION: Different than script for v1!

clear all;

% OUTPUT
pointclouds_dir = 'pointclouds';
lbl_pointclouds_dir = 'labeled_pointclouds';
cameraangles_file = 'cam_angles.txt';
trainingindex_file = 'indextraining.txt';
testindex_file = 'indextest.txt';
colorcode_file = 'color_code.txt';

savelabels = 1;         % save labeled pointclouds?
savepointclouds = 1;    % save rgb pointclouds?
savecameraangles = 1;   % save camera angles file?
savelabelnames = 1;     % save label names file?
saveindexfiles = 1;     % save index files?

dataset_file = 'nyu_depth_v2_labeled.mat';
split_file = 'splits.mat';  % downloaded if necessary
accel_calib_file = 'accel_calib_for_daniel.mat';  % obtained directly from Silberman
labelnames_file = 'labelnames.txt';

if (savelabels || savepointclouds || savecameraangles) && ~exist(dataset_file, 'file')
    fprintf(1, 'Downloading dataset...');
    urlwrite('https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/v2/nyu_depth_v2_labeled.mat', dataset_file);
    fprintf(1, 'Done\n');
end
if savecameraangles && ~exist(accel_calib_file, 'file')
    fprintf(1, 'Downloading accelerometer calibration...');    
    urlwrite('https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/accel_calib_for_daniel.mat', accel_calib_file);
    fprintf(1, 'Done\n');
end
if saveindexfiles
    fprintf(1, 'Downloading train/test split files...');    
    if ~exist(split_file, 'file')
        urlwrite('https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/v2/splits.mat', split_file);
    end    
    fprintf(1, 'Done\n');
end
if ~exist(colorcode_file, 'file')
    fprintf(1, 'Downloading color code file...');        
    urlwrite('https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/v2/color_code.txt', colorcode_file);
    fprintf(1, 'Done\n');
end
if ~exist(labelnames_file, 'file')
    fprintf(1, 'Downloading label names file...');        
    urlwrite('https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/v2/labelnames.txt', labelnames_file);
    fprintf(1, 'Done\n');
end

% compile uwrite code
mex -outdir uwrite uwrite/uwrite.c

addpath('mat2pcl');
addpath('uwrite');
addpath('toolbox');

% load couprie label mapping
couprie_labeling;

fprintf(1, 'Load dataset...\n');

if savepointclouds && ~exist('images','var')
    load(dataset_file, 'images');
end
if (savepointclouds || savelabels) && ~exist('rawDepths','var')
    load(dataset_file, 'rawDepths');
end
if savelabels && ~exist('labels', 'var')
    load(dataset_file, 'labels');
end

if savepointclouds || savelabels
    fprintf(1, 'Convert point clouds...\n');
    
    rows = size(images,1);
    cols = size(images,2);
    nPointClouds = size(labels,3);

    fprintf(1, 'Start matlabpool for parallel processing...\n');

    if matlabpool('SIZE') == 0
        matlabpool(8);
    end

    mkdir(pointclouds_dir);
    mkdir(lbl_pointclouds_dir);

    % replace invalid values with NaN
    rawDepths(rawDepths == 0) = NaN;

    parfor i=1:40    
        fprintf(1,'Create %04d.pcd\n', i);

        depth_world = rgb_plane2rgb_world(rawDepths(:,:,i));
        rgb = single(images(:,:,:,i))./255;    
        lbls = single(maplabels(labels(:,:,i), couprie));

        depth_world = reshape(depth_world, rows, cols, 3);

        %%%%%%% SAVE PCD FILE %%%%%%
        if savepointclouds
            pcd = cat(3, depth_world, rgb);
            filename = sprintf('%s/%04d',pointclouds_dir, i);
            savepcd([filename '.pcd'], pcd, 'binary');
        end

        %%%%%% SAVE LABEL PCD FILE %%%%%%
        if savelabels        
            pcd = cat(3, depth_world, lbls);
            filename = sprintf('%s/%04d', lbl_pointclouds_dir, i);
            savexyzl([filename '.pcd'], pcd);
        end       
    end

    matlabpool('CLOSE');
    fprintf(1, 'DONE\n');
end

if savecameraangles
    fprintf(1, 'Save camera angles...');
    
    load(dataset_file, 'accelData');
    load(accel_calib_file,'pitch_calib', 'roll_calib');
    
    % roll angle
    r = interp1(roll_calib(:,2), roll_calib(:,1), accelData(:,1), 'cubic');
    % pitch angle
    p = interp1(pitch_calib(:,2), pitch_calib(:,1), accelData(:,3), 'cubic');

    a = deg2rad(double([r p]));
    save(cameraangles_file, 'a', '-ascii')
    
    fprintf(1, 'DONE\n');
end

if savelabelnames
    fprintf(1, 'Save label names...');
    mappedlabelnames = {'bed', 'object', 'chair', 'furniture', 'ceiling', 'floor', 'deco', 'sofa', 'table', 'wall', 'window', 'bookshelf', 'tv'};
    f = fopen(labelnames_file, 'w+');
    fprintf(f, '%s\n', mappedlabelnames{:});
    fclose(f);
    fprintf(1, 'DONE\n');
end

if saveindexfiles
    fprintf(1, 'Save index files...');
    load(split_file, 'trainNdxs', 'testNdxs');
    
    f = fopen(trainingindex_file, 'w+');
    fprintf(f, '%04d\n', trainNdxs);
    fclose(f);
    
    f = fopen(testindex_file, 'w+');
    fprintf(f, '%04d\n', testNdxs);
    fclose(f);
    fprintf(1, 'DONE\n');
end