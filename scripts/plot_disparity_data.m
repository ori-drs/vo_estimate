%0 far/none
%7 long range threshold

%45 disparity short range threshold -----
%60 short
%120 disparity short

% low threshold of 0.7
% short range of 0.82 is ground at feet
% long range of 
% high threshold of 4m
% end of room 20m 

% = focal_length * baseline/50;

close all, clear all
x= load('example.txt');

imagesc(x), colorbar
title('orig')
keyboard

b_low = 7 % 5.53
b_high = 45 % 0.8613

x(x < b_low) = 0;
x(x > b_high) = 0;
imagesc(x,[b_low b_high]), colorbar

focal_length = 553.6858520507812
baseline = 0.07

d = focal_length * baseline ./ x;




figure; imagesc(d), colorbar