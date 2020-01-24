% Location of the compressed data set
url = 'http://download.tensorflow.org/example_images/flower_photos.tgz';

% Store the output in a temporary folder
downloadFolder = tempdir;
filename = fullfile(downloadFolder,'flower_dataset.tgz');
% Uncompressed data set
imageFolder = fullfile(downloadFolder,'flower_photos');

if ~exist(imageFolder,'dir') % download only once
    disp('Downloading Flower Dataset (218 MB)...');
    websave(filename,url);
    untar(filename,downloadFolder)
end
% Display a one of the flower images
figure
I = imread(flowerImageSet.Files{1});
imshow(I);
flowerImageSet = imageDatastore(imageFolder,'LabelSource','foldernames','IncludeSubfolders',true);
% Total number of images in the data set
numel(flowerImageSet.Files)
% Display a one of the flower images
figure
I = imread(flowerImageSet.Files{1});
imshow(I);

function [features, metrics] = exampleBagOfFeaturesColorExtractor(I) 
% Example color layout feature extractor. Designed for use with bagOfFeatures.
%
% Local color layout features are extracted from truecolor image, I and
% returned in features. The strength of the features are returned in
% metrics.

[~,~,P] = size(I);

isColorImage = P == 3; 

if isColorImage
    
    % Convert RGB images to the L*a*b* colorspace. The L*a*b* colorspace
    % enables you to easily quantify the visual differences between colors.
    % Visually similar colors in the L*a*b* colorspace will have small
    % differences in their L*a*b* values.
    Ilab = rgb2lab(I);                                                                             
      
    % Compute the "average" L*a*b* color within 16-by-16 pixel blocks. The
    % average value is used as the color portion of the image feature. An
    % efficient method to approximate this averaging procedure over
    % 16-by-16 pixel blocks is to reduce the size of the image by a factor
    % of 16 using IMRESIZE. 
    Ilab = imresize(Ilab, 1/16);
    
    % Note, the average pixel value in a block can also be computed using
    % standard block processing or integral images.
    
    % Reshape L*a*b* image into "number of features"-by-3 matrix.
    [Mr,Nr,~] = size(Ilab);    
    colorFeatures = reshape(Ilab, Mr*Nr, []); 
           
    % L2 normalize color features
    rowNorm = sqrt(sum(colorFeatures.^2,2));
    colorFeatures = bsxfun(@rdivide, colorFeatures, rowNorm + eps);
        
    % Augment the color feature by appending the [x y] location within the
    % image from which the color feature was extracted. This technique is
    % known as spatial augmentation. Spatial augmentation incorporates the
    % spatial layout of the features within an image as part of the
    % extracted feature vectors. Therefore, for two images to have similar
    % color features, the color and spatial distribution of color must be
    % similar.
    
    % Normalize pixel coordinates to handle different image sizes.
    xnorm = linspace(-0.5, 0.5, Nr);      
    ynorm = linspace(-0.5, 0.5, Mr);    
    [x, y] = meshgrid(xnorm, ynorm);
    
    % Concatenate the spatial locations and color features.
    features = [colorFeatures y(:) x(:)];
    
    % Use color variance as feature metric.
    metrics  = var(colorFeatures(:,1:3),0,2);
else
    
    % Return empty features for non-color images. These features are
    % ignored by bagOfFeatures.
    features = zeros(0,5);
    metrics  = zeros(0,1);     
end
  doTraining = true;

if doTraining
    %Pick a random subset of the flower images
    trainingSet = splitEachLabel(flowerImageSet, 0.6, 'randomized');
    
    % Create a custom bag of features using the 'CustomExtractor' option
    colorBag = bagOfFeatures(trainingSet, ...
        'CustomExtractor', @exampleBagOfFeaturesColorExtractor, ...
        'VocabularySize', 5000);
%else
    % Load a pretrained bagOfFeatures
    load('savedColorBagOfFeatures.mat','colorBag');
%end
if doTraining
    % Create a search index
    flowerImageIndex = indexImages(flowerImageSet,colorBag,'SaveFeatureLocations',true);

    % Load aif doTraining
    % Create a search index
else
    % Load a saved index
    load('savedColorBagOfFeatures.mat','flowerImageIndex');


end

% Define a query image% Define a query image
queryImage = readimage(flowerImageSet,200);
figure
imshow(queryImage)




[imageIDs, scores] = retrieveImages(queryImage, flowerImageIndex,'NumResults',4);
 
montage(flowerImageSet.Files(imageIDs),'ThumbnailSize',[200 200])
end
end
