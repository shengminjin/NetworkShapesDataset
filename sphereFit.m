function [center, radius] = sphereFit(points)
    center = mean(points);
    radius = 0;
    for j = 1:length(points)
        radius = max(radius, norm(points(j,:) - center));
    end