% This script converts the labeled RGBD frames of NYUv1 into pcd pointcloud
% files readable by PCL.
% Note: groundtruth labels are already mapped (all points not labeled from
% 1-12 are mapped background (13)
% ATTENTION: Different than script for v2!

clear all;

% OUTPUT DIRECTORIES
pointclouds_dir = 'pointclouds';
lbl_pointclouds_dir = 'labeled_pointclouds';
cameraangles_file = 'cam_angles.txt';
trainingindex_file = 'indextraining';
testindex_file = 'indextest';
colorcode_file = 'color_code.txt';

savelabels = 0;         % save labeled pointclouds?
savepointclouds = 0;    % save rgb pointclouds?
savecameraangles = 0;   % save camera angles file?
savelabelnames = 0;     % save label names file?
saveindexfiles = 1;     % save index files?

dataset_file = 'nyu_depth_data_labeled.mat';
splitfile_prefix = 'splits_fold';
accel_calib_file = 'accel_calib_for_daniel.mat';  % obtained directly from Silberman
labelnames_file = 'labelnames.txt';

if (savelabels || savepointclouds || savecameraangles) && ~exist(dataset_file, 'file')
    fprintf(1, 'Downloading dataset...');
    urlwrite('https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/v1/nyu_depth_data_labeled.mat', dataset_file);
    fprintf(1, 'Done\n');
end
if savecameraangles && ~exist(accel_calib_file, 'file')
    fprintf(1, 'Downloading accelerometer calibration...');    
    urlwrite('https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/accel_calib_for_daniel.mat', accel_calib_file);
    fprintf(1, 'Done\n');
end
if saveindexfiles
    fprintf(1, 'Downloading train/test split files...');
    for i=1:10
        splitfile = [splitfile_prefix num2str(i) '.mat'];
        
        if ~exist(splitfile, 'file')
            urlwrite(['https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/v1/' splitfile], splitfile);
        end
    end
    fprintf(1, 'Done\n');
end
if ~exist(colorcode_file, 'file')
    fprintf(1, 'Downloading color code file...');        
    urlwrite('https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/v1/color_code.txt', colorcode_file);
    fprintf(1, 'Done\n');
end
if ~exist(labelnames_file, 'file')
    fprintf(1, 'Downloading label names file...');        
    urlwrite('https://repo.acin.tuwien.ac.at/tmp/permanent/NYU_Depth_Dataset/v1/labelnames.txt', labelnames_file);
    fprintf(1, 'Done\n');
end

% compile uwrite code
mex -outdir uwrite uwrite/uwrite.c

addpath('mat2pcl');
addpath('uwrite');
addpath('toolbox');

% load nyuv1 label mapping
nyuv1_mapping;

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

    parfor i=1:nPointClouds   
        fprintf(1,'Create %04d.pcd\n', i);
        depth_rgb_frame = get_projected_depth(swapbytes(rawDepths(:,:,i)));
        depth_rgb_frame(depth_rgb_frame == 0 | depth_rgb_frame == 10) = NaN;

        rgb = single(images(:,:,:,i))./255;    
        lbls = single(maplabels(labels(:,:,i), v1_mapping));

        depth_world = rgb_plane2rgb_world(depth_rgb_frame);    
        depth_world = reshape(depth_world, rows, cols, 3);

        %%%%%% SAVE PCD FILE %%%%%%
        if savepointclouds
            pcd = cat(3, depth_world, rgb);
            filename = sprintf('%s/%04d.pcd',pointclouds_dir, i);
            savepcd(filename, pcd, 'binary');
        end

        %%%%%% SAVE LABEL PCD FILE %%%%%%
        if savelabels        
            pcd = cat(3, depth_world, lbls);
            filename = sprintf('%s/%04d.pcd',lbl_pointclouds_dir, i);
            savexyzl_fromv2(filename, pcd);
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
    mappedlabelnames = {'bed','blind','bookshelf','cabinet','ceiling','floor','picture','sofa','table','tv','wall','window','background'};
    f = fopen(labelnames_file, 'w+');
    fprintf(f, '%s\n', mappedlabelnames{:});
    fclose(f);
    fprintf(1, 'DONE\n');
end

if saveindexfiles
    fprintf(1, 'Save index files...');
    
    for i=1:10
        load([splitfile_prefix num2str(i) '.mat'], 'trainNdxs', 'testNdxs');

        f = fopen([trainingindex_file num2str(i) '.txt'], 'w+');
        fprintf(f, '%04d\n', trainNdxs);
        fclose(f);

        f = fopen([testindex_file num2str(i) '.txt'], 'w+');
        fprintf(f, '%04d\n', testNdxs);
        fclose(f);
    end
    
    fprintf(1, 'DONE\n');
end