function savexyzl(fname, points)
    % save points in xyz format
    % TODO
    %  binary format, RGB
    
   fp = fopen(fname, 'w');
    
    % find the attributes of the point cloud

    % old, only for unorganized
%     npoints = size(points, 2);
%     width = npoints;
%     height  = 1;
%     nfields = size(points, 1);
         
    
    if ndims(points) == 2
        % unorganized point cloud
        npoints = size(points, 2);
        width = npoints;
        height  = 1;
        nfields = size(points, 1);
    else
        width = size(points, 2);
        height  = size(points, 1);
        npoints = width*height;
        nfields = size(points, 3);

        % put the data in order with one column per point
        points = permute(points, [2 1 3]);
        points = reshape(points, [], size(points,3))';        
    end
    

    fieldstr = 'x y z label';
    count = '1 1 1 1';
  
    typ = 'F F F U';
  
    siz = '4 4 4 4';
       
    
    % write the PCD file header
    
    fprintf(fp, '# .PCD v.7 - Point Cloud Data file format\n');
    fprintf(fp, 'VERSION .7\n');
    
    fprintf(fp, 'FIELDS %s\n', fieldstr);
    fprintf(fp, 'SIZE %s\n', siz);
    fprintf(fp, 'TYPE %s\n', typ);
    fprintf(fp, 'COUNT %s\n', count);
        
    fprintf(fp, 'WIDTH %d\n', width);
    fprintf(fp, 'HEIGHT %d\n', height);
    fprintf(fp, 'POINTS %d\n', npoints);
    

    lbl = double(points(4,:));
    %points = [ points(1:3,:); double(typecast(lbl,'uint32'))];
       
    
   
    % Write binary format data
    fprintf(fp, 'DATA binary\n');

    % for a full color point cloud the colors are not quite right in pclviewer,
    % color as a float has only 23 bits of mantissa precision, not enough for
    % RGB as 8 bits each

    % write color as a float not an int
%     for i=1:npoints      
%         fwrite(fp, points(1:3,i), 'float32');
%         fwrite(fp, uint32(points(4,i)), 'integer*4');     
%     end

    A = uwrite(double(points(1:3,:)), 'float32');
    B = reshape(A, 12, length(A)/12);
    C = uwrite(double(points(4,:)), 'uint32');
    D = reshape(C, 4, length(C)/4);
    E = [B;D];
    fwrite(fp, E, 'uint8');
    fclose(fp);
end


    

