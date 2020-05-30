function p = get_sphere(points, directory, display_name)

%    points = points{1};
    warning('off');
    fig = figure(1);
    X = double(points(:, 1));
    Y = double(points(:, 2));
    Z = double(points(:, 3));
    C = double(points(:, 4));

    [center, radius] = sphereFit(points(:, 1:3));
    [x,y,z] = sphere(10);
    surf(x*radius(1)+center(1),y*radius(1)+center(2),z*radius(1)+center(3), 'Facecolor', 'r')

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
    fig_file = strcat(directory, '/sphere.fig');
    savefig(fig_file)
    png_file = strcat(directory, '/sphere.png');
    saveas(fig, png_file)

    boundary_file = strcat(directory, '/center_radius.txt');
    tmp = [center, radius];
    writematrix(tmp, boundary_file)
end
