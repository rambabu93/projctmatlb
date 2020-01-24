
% This code implemented license plate detection using morphological operators and loosely follows the approach presented by:
% 1. Farhad Faradji, Amir Hossein Rezaie, Majid Ziaratban a morphological based license plate location ICIP 2007.
% 2. Farhad Faraji, and Reza Safabakhsh, novel and fast method for detecting plate location from complicated images based on morphological 
% operations MVIP 2007.

% The steps are as follows:
% 1. Vertical edge detection
% 2. Histogram analysis
% 3. Vertical and Horizontal Dilation
% 4. Finding Regions in Common
% 5. Horizontal Dilation
% 6. Erosion
% 7. Post processing

% Alireza Asvadi
% Department of ECE, SPR Lab
% Babol (Noshirvani) University of Technology
% http://www.a-asvadi.ir
% 2012
%% clear command windows
clc
clear all
close all
%% Read Image
In    = 1;                              % select a number between 1 to 9
Im    = imread('car4.jpg'); % read image
I     = im2double(rgb2gray(Im));        % rgb to gray
% figure();imshow(I)
%% Sobel Masking 
SM    = [-1 0 1;-2 0 2;-1 0 1];         % Sobel Vertical Mask
IS    = imfilter(I,SM,'replicate');     % Filter Image Using Sobel Mask
IS    = IS.^2;                          % Consider Just Value of Edges & Fray Weak Edges
 figure();imshow(IS)
%% Normalization
IS    = (IS-min(IS(:)))/(max(IS(:))-min(IS(:))); % Normalization
figure();imshow(IS)
%% Threshold (Otsu)
%level = graythresh(IS);                 % Threshold Based on Otsu Method
%IS    = im2bw(IS,level);
%figure();imshow(IS)
level = graythresh(IS);                 % Threshold Based on Otsu Method

SA    = imbinarize(IS);
figure();
imshow(SA)
imshow(level)
IS    = im2bw(IS,level);

%% Histogram
S     = sum(IS,2);                      % Edge Horizontal Histogram
 figure();plot(1:size(S,1),S)
 view(90,90)
%% Plot
 figure()
 subplot(1,2,1);imshow(IS)
subplot(1,2,2);plot(1:size(S,1),S)
axis([1 size(IS,1) 0 max(S)]);view(90,90)
%% Plate Location
T1    = 0.35;                           % Threshold On Edge Histogram
PR    = find(S > (T1*max(S)));          % Candidate Plate Rows
%% Masked Plate
Msk   = zeros(size(I));
Msk(PR,:) = 1;                          % Mask
MB    = Msk.*IS;                        % Candidate Plate (Edge Image)% figure();imshow(MB)
%% Morphology (Dilation - Vertical)
Dy    = strel('rectangle',[80,4]);      % Vertical Extension
%Dy    = strel('rectangle',[80,4]);      % Vertical Extension

MBy   = imdilate(MB,Dy);                % By Dilation
MBy   = imfill(MBy,'holes');            % Fill Holes
 figure();imshow(MBy)
%% Morphology (Dilation - Horizontal)
Dx    = strel('rectangle',[4,80]);      % Horizontal Extension
MBx   = imdilate(MB,Dx);                % By Dilation
MBx   = imfill(MBx,'holes');            % Fill Holes
 figure();imshow(MBy)
%% Joint Places
BIM   = MBx.*MBy;                       % Joint Places
 figure();imshow(BIM)
%% Morphology (Dilation - Horizontal)
Dy    = strel('rectangle',[4,30]);      % Horizontal Extension
MM    = imdilate(BIM,Dy);               % By Dilation
MM    = imfill(MM,'holes');             % Fill Holes
 figure();imshow(MM)
%% Erosion
Dr    = strel('line',50,0);             % Erosion
BL    = imerode(MM,Dr);
figure();imshow(BL)
%% Find Biggest Binary Region (As a Plate Place)
[L,num] = bwlabel(BL);                  % Label (Binary Regions)               
Areas   = zeros(num,1);
for i = 1:num                           % Compute Area Of Every Region
[r,c,v]  = find(L == i);                % Find Indexes
Areas(i) = sum(v);                      % Compute Area    
end 
[La,Lb] = find(Areas==max(Areas));      % Biggest Binary Region Index
%% Post Processing
[a,b]   = find(L==La);                  % Find Biggest Binary Region (Plate)
[nRow,nCol] = size(I);
FM      = zeros(nRow,nCol);             % Smooth and Enlarge Plate Place
T       = 10;                           % Extend Plate Region By T Pixel
jr      = (min(a)-T :max(a)+T);
jc      = (min(b)-T :max(b)+T);
jr      = jr(jr >= 1 & jr <= nRow);
jc      = jc(jc >= 1 & jc <= nCol);
FM(jr,jc) = 1; 
PL      = FM.*I;                        % Detected Plate
 figure();imshow(FM)
 figure();imshow(PL)
%% Plot
imshow(Im); title('Detected Plate')
hold on
rectangle('Position',[min(jc),min(jr),max(jc)-min(jc),...
max(jr)-min(jr)],'LineWidth',4,'EdgeColor','r')
hold off
