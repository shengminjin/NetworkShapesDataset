function p = get_cuboid(points, directory, display_name)
    warning('off');
    try
        addpath(genpath('minboundbox'))
        fig = figure(1);
        X = double(points(:, 1));
        Y = double(points(:, 2));
        Z = double(points(:, 3));
        C = double(points(:, 4));

        [rotmat,cornerpoints,volume,surface,edgelength] = minboundbox(X,Y,Z);
        shading faceted
        [k, v] = convhull(cornerpoints(:, 1), cornerpoints(:, 2), cornerpoints(:, 3), 'simplify', true);
        trisurf(k, cornerpoints(:, 1), cornerpoints(:, 2), cornerpoints(:, 3),'Facecolor','r')

        hold on
        scatter3(X, Y, Z, [12], C)

        p = 1;
        xlabel('a')
        ylabel('b')
        zlabel('d')
        alpha 0.1
    %    legend
        hcb=colorbar;
        title(hcb,'Sampling Proportion');

        hold off
        fig_file = strcat(directory, '/cuboid.fig');
        savefig(fig_file)
        png_file = strcat(directory, '/cuboid.png');
        saveas(fig, png_file)

        boundary_file = strcat(directory, '/corner_points.txt');
        writematrix(cornerpoints, boundary_file);
    catch
        p = 0;
        error_message = "Error computing the cuboid/convex hull. The points may be coplanar or collinear. Please see the kron_points.txt for the points.";
        error_log = strcat(directory, '/error.log');
        fid = fopen(error_log,'wt');
        fprintf(fid, string(error_message));
        fclose(fid);
    end
end
