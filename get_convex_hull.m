function p = get_convex_hull(points, directory, display_name)
    warning('off');
    fig = figure(1);
    X = double(points(:, 1));
    Y = double(points(:, 2));
    Z = double(points(:, 3));
    C = double(points(:, 4));

    try
        [k, v] = convhull(X, Y, Z, 'simplify', true);

        trisurf(k, X, Y, Z, 'Facecolor', 'r', 'DisplayName', string(display_name));
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
        fig_file = strcat(directory, '/convexhull.fig');
        savefig(fig_file)
        png_file = strcat(directory, '/convexhull.png');
        saveas(fig, png_file)

        boundary_file = strcat(directory, '/boundary.txt');
        vertX = reshape(X(k), [size(k, 1)*size(k, 2), 1]);
        vertY = reshape(Y(k), [size(k, 1)*size(k, 2), 1]);
        vertZ = reshape(Z(k), [size(k, 1)*size(k, 2), 1]);
        vertices = unique(cat(2, cat(2, vertX, vertY), vertZ), 'rows');
        writematrix(vertices, boundary_file);
    catch
        p = 0;
        error_message = "Error computing the cuboid/convex hull. The points may be coplanar or collinear. Please see the kron_points.txt for the points.";
        error_log = strcat(directory, '/error.log');
        fid = fopen(error_log,'wt');
        fprintf(fid, string(error_message));
        fclose(fid);
    end
end
