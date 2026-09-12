clc;
clear;
close all;

fprintf('\n============================================\n');
fprintf('   DIABETIC RETINOPATHY DETECTION USING SVM\n');
fprintf('============================================\n\n');

projectFolder = fileparts(mfilename('fullpath'));
datasetFolder = fullfile(projectFolder,'Datasets','Dataset 2','images');

if ~isfolder(datasetFolder)
    error('Dataset 2/images folder was not found.');
end

extensions = {'*.jpg','*.JPG','*.jpeg','*.JPEG','*.png','*.PNG','*.bmp','*.BMP'};
files = [];

for e = 1:length(extensions)
    temp = dir(fullfile(datasetFolder,extensions{e}));
    if ~isempty(temp)
        files = [files; temp];
    end
end

if isempty(files)
    error('No images were found in Dataset 2/images.');
end

filePaths = cell(length(files),1);
labels = zeros(length(files),1);

for i = 1:length(files)

    filePaths{i} = fullfile(datasetFolder,files(i).name);

    name = lower(files(i).name);

    if contains(name,'_dr')
        labels(i) = 1;
    elseif contains(name,'_g') || contains(name,'_h')
        labels(i) = 0;
    else
        labels(i) = -1;
    end
end

valid = labels ~= -1;

filePaths = filePaths(valid);
labels = labels(valid);

fprintf('Valid images: %d\n',length(labels));
fprintf('DR images: %d\n',sum(labels == 1));
fprintf('No-DR images: %d\n\n',sum(labels == 0));

fprintf('Extracting features...\n');

numImages = length(filePaths);
firstFeature = extractFeaturesNoToolbox(filePaths{1});
numFeatures = length(firstFeature);

X = zeros(numImages,numFeatures);

for i = 1:numImages

    X(i,:) = extractFeaturesNoToolbox(filePaths{i});

    if mod(i,10) == 0 || i == numImages
        fprintf('Processed %d/%d images\n',i,numImages);
    end
end

fprintf('\nFeature extraction completed.\n');
fprintf('Number of features: %d\n\n',numFeatures);

rng(42);

drIndex = find(labels == 1);
noDrIndex = find(labels == 0);

drIndex = drIndex(randperm(length(drIndex)));
noDrIndex = noDrIndex(randperm(length(noDrIndex)));

nDrTrain = max(1,round(0.70*length(drIndex)));
nNoDrTrain = max(1,round(0.70*length(noDrIndex)));

trainIndex = [drIndex(1:nDrTrain); noDrIndex(1:nNoDrTrain)];

testIndex = [drIndex(nDrTrain+1:end); noDrIndex(nNoDrTrain+1:end)];

trainIndex = trainIndex(randperm(length(trainIndex)));
testIndex = testIndex(randperm(length(testIndex)));

XTrain = X(trainIndex,:);
YTrain = labels(trainIndex);

XTest = X(testIndex,:);
YTest = labels(testIndex);

mu = mean(XTrain,1);
sigmaFeature = std(XTrain,0,1);

sigmaFeature(sigmaFeature < 1e-12) = 1;

XTrain = (XTrain - mu) ./ sigmaFeature;
XTest = (XTest - mu) ./ sigmaFeature;

fprintf('Training images: %d\n',length(YTrain));
fprintf('Testing images: %d\n\n',length(YTest));

fprintf('Training SVM...\n');

YTrainSVM = -ones(length(YTrain),1);
YTrainSVM(YTrain == 1) = 1;

svmModel = trainRBFSVM(XTrain,YTrainSVM,10);

svmModel.mu = mu;
svmModel.sigmaFeature = sigmaFeature;
svmModel.featureCount = numFeatures;

fprintf('SVM training completed.\n\n');

scores = svmPredict(XTest,svmModel);

predictedLabels = zeros(length(scores),1);
predictedLabels(scores >= 0) = 1;

TP = sum((predictedLabels == 1) & (YTest == 1));
TN = sum((predictedLabels == 0) & (YTest == 0));
FP = sum((predictedLabels == 1) & (YTest == 0));
FN = sum((predictedLabels == 0) & (YTest == 1));

accuracy = (TP + TN) / max(1,length(YTest));
sensitivity = TP / max(1,TP + FN);
specificity = TN / max(1,TN + FP);

fprintf('============================================\n');
fprintf('             MODEL PERFORMANCE\n');
fprintf('============================================\n');
fprintf('Accuracy    : %.2f %%\n',accuracy*100);
fprintf('Sensitivity : %.2f %%\n',sensitivity*100);
fprintf('Specificity : %.2f %%\n',specificity*100);
fprintf('============================================\n\n');

fprintf('Confusion Matrix:\n');
fprintf('\n');
fprintf('                 Actual\n');
fprintf('              No DR    DR\n');
fprintf('Pred No DR     %4d   %4d\n',TN,FN);
fprintf('Pred DR        %4d   %4d\n\n',FP,TP);

figure('Name','DR Confusion Matrix','NumberTitle','off');

confMatrix = [TN FP; FN TP];

imagesc(confMatrix);
axis equal;
axis tight;

title('Diabetic Retinopathy Confusion Matrix');
xlabel('Predicted Class');
ylabel('Actual Class');

set(gca,'XTick',[1 2]);
set(gca,'XTickLabel',{'No DR','DR'});
set(gca,'YTick',[1 2]);
set(gca,'YTickLabel',{'No DR','DR'});

for r = 1:2
    for c = 1:2
        text(c,r,num2str(confMatrix(r,c)), ...
            'HorizontalAlignment','center', ...
            'FontSize',14, ...
            'FontWeight','bold');
    end
end

save(fullfile(projectFolder,'DR_SVM_Model.mat'),'svmModel');

fprintf('Model saved as DR_SVM_Model.mat\n\n');

choice = input('Do you want to test a new fundus image? (Y/N): ','s');

if strcmpi(choice,'Y')

    [selectedFile,selectedPath] = uigetfile( ...
        {'*.jpg;*.JPG;*.jpeg;*.JPEG;*.png;*.PNG;*.bmp;*.BMP', ...
        'Fundus Images'}, ...
        'Select a Fundus Image');

    if isequal(selectedFile,0)

        fprintf('\nNo image selected.\n');

    else

        testImagePath = fullfile(selectedPath,selectedFile);

        featureVector = extractFeaturesNoToolbox(testImagePath);

        featureVector = (featureVector - svmModel.mu) ...
            ./ svmModel.sigmaFeature;

        resultScore = svmPredict(featureVector,svmModel);

        figure('Name','Fundus Image','NumberTitle','off');

        originalImage = imread(testImagePath);

        imshow(originalImage);
        title('Selected Fundus Image');

        fprintf('\n============================================\n');
        fprintf('              SCREENING RESULT\n');
        fprintf('============================================\n');

        if resultScore >= 0

            fprintf('RESULT: DIABETIC RETINOPATHY DETECTED\n');
            fprintf('The model found features associated with DR.\n');

        else

            fprintf('RESULT: NO DR DETECTED\n');
            fprintf('The model did not find features associated with DR.\n');

        end

        fprintf('============================================\n\n');
        fprintf('This is an AI screening result, not a medical diagnosis.\n');

    end
end


function featureVector = extractFeaturesNoToolbox(imagePath)

    img = imread(imagePath);

    if ndims(img) == 3

        img = double(img);

        green = img(:,:,2);

        gray = 0.299*img(:,:,1) + ...
               0.587*img(:,:,2) + ...
               0.114*img(:,:,3);

    else

        gray = double(img);
        green = gray;

    end

    green = normalizeImage(green);

    green = localContrastEnhancement(green);

    resizedGreen = resizeImage(green,64,64);

    lbpFeature = calculateLBP(resizedGreen);

    glcmFeature = calculateGLCMFeatures(resizedGreen);

    featureVector = [lbpFeature glcmFeature];

end


function output = normalizeImage(inputImage)

    inputImage = double(inputImage);

    minValue = min(inputImage(:));
    maxValue = max(inputImage(:));

    if maxValue - minValue < 1e-12
        output = zeros(size(inputImage));
    else
        output = (inputImage - minValue) ...
            /(maxValue - minValue);
    end

end


function output = localContrastEnhancement(inputImage)

    inputImage = normalizeImage(inputImage);

    kernelSize = 15;

    kernel = ones(kernelSize,kernelSize) ...
        /(kernelSize*kernelSize);

    localMean = conv2(inputImage,kernel,'same');

    localSquaredMean = conv2(inputImage.^2,kernel,'same');

    localVariance = localSquaredMean - localMean.^2;

    localVariance(localVariance < 0) = 0;

    localStd = sqrt(localVariance);

    output = (inputImage - localMean) ...
        ./(localStd + 0.05);

    output = 0.5 + 0.25*output;

    output(output < 0) = 0;
    output(output > 1) = 1;

end


function output = resizeImage(inputImage,newRows,newCols)

    [rows,cols] = size(inputImage);

    xOld = 1:cols;
    yOld = 1:rows;

    xNew = linspace(1,cols,newCols);
    yNew = linspace(1,rows,newRows);

    [XOld,YOld] = meshgrid(xOld,yOld);
    [XNew,YNew] = meshgrid(xNew,yNew);

    output = interp2(XOld,YOld,inputImage, ...
        XNew,YNew,'linear');

    output(isnan(output)) = 0;

end


function feature = calculateLBP(image)

    image = normalizeImage(image);

    image = round(image*255);

    [rows,cols] = size(image);

    histogramValues = zeros(1,256);

    for r = 2:rows-1

        for c = 2:cols-1

            center = image(r,c);

            code = 0;

            if image(r-1,c-1) >= center
                code = code + 1;
            end

            if image(r-1,c) >= center
                code = code + 2;
            end

            if image(r-1,c+1) >= center
                code = code + 4;
            end

            if image(r,c+1) >= center
                code = code + 8;
            end

            if image(r+1,c+1) >= center
                code = code + 16;
            end

            if image(r+1,c) >= center
                code = code + 32;
            end

            if image(r+1,c-1) >= center
                code = code + 64;
            end

            if image(r,c-1) >= center
                code = code + 128;
            end

            histogramValues(code+1) = ...
                histogramValues(code+1) + 1;

        end

    end

    total = sum(histogramValues);

    if total > 0
        feature = histogramValues / total;
    else
        feature = histogramValues;
    end

end


function feature = calculateGLCMFeatures(image)

    image = normalizeImage(image);

    levels = 16;

    quantized = floor(image*(levels-1)) + 1;

    [rows,cols] = size(quantized);

    offsets = [0 1; -1 1; -1 0; -1 -1];

    feature = zeros(1,16);

    position = 1;

    for direction = 1:4

        dr = offsets(direction,1);
        dc = offsets(direction,2);

        glcm = zeros(levels,levels);

        for r = 1:rows

            for c = 1:cols

                r2 = r + dr;
                c2 = c + dc;

                if r2 >= 1 && r2 <= rows && ...
                   c2 >= 1 && c2 <= cols

                    a = quantized(r,c);
                    b = quantized(r2,c2);

                    glcm(a,b) = glcm(a,b) + 1;

                end

            end

        end

        glcm = glcm + glcm';

        total = sum(glcm(:));

        if total > 0
            glcm = glcm / total;
        end

        contrast = 0;
        homogeneity = 0;
        energy = 0;
        correlation = 0;

        meanI = 0;
        meanJ = 0;

        for i = 1:levels
            for j = 1:levels

                probability = glcm(i,j);

                meanI = meanI + i*probability;
                meanJ = meanJ + j*probability;

            end
        end

        stdI = 0;
        stdJ = 0;

        for i = 1:levels
            for j = 1:levels

                probability = glcm(i,j);

                stdI = stdI + ((i-meanI)^2)*probability;
                stdJ = stdJ + ((j-meanJ)^2)*probability;

            end
        end

        stdI = sqrt(stdI);
        stdJ = sqrt(stdJ);

        for i = 1:levels
            for j = 1:levels

                probability = glcm(i,j);

                contrast = contrast + ...
                    ((i-j)^2)*probability;

                homogeneity = homogeneity + ...
                    probability/(1+abs(i-j));

                energy = energy + probability^2;

                if stdI > 1e-12 && stdJ > 1e-12

                    correlation = correlation + ...
                        ((i-meanI)*(j-meanJ)*probability) ...
                        /(stdI*stdJ);

                end

            end
        end

        energy = sqrt(energy);

        feature(position) = contrast;
        feature(position+1) = homogeneity;
        feature(position+2) = energy;
        feature(position+3) = correlation;

        position = position + 4;

    end

end


function model = trainRBFSVM(X,Y,C)

    n = size(X,1);

    gamma = calculateGamma(X);

    K = rbfKernel(X,X,gamma);

    alpha = zeros(n,1);

    bias = 0;

    tolerance = 1e-3;

    maxPasses = 15;

    passes = 0;

    maxIterations = 20000;

    iteration = 0;

    while passes < maxPasses && iteration < maxIterations

        changed = 0;

        for i = 1:n

            Ei = sum(alpha .* Y .* K(:,i)) ...
                + bias - Y(i);

            condition1 = ...
                (Y(i)*Ei < -tolerance && alpha(i) < C);

            condition2 = ...
                (Y(i)*Ei > tolerance && alpha(i) > 0);

            if condition1 || condition2

                j = i;

                while j == i
                    j = randi(n);
                end

                Ej = sum(alpha .* Y .* K(:,j)) ...
                    + bias - Y(j);

                oldAi = alpha(i);
                oldAj = alpha(j);

                if Y(i) ~= Y(j)

                    L = max(0,alpha(j)-alpha(i));
                    H = min(C,C+alpha(j)-alpha(i));

                else

                    L = max(0,alpha(i)+alpha(j)-C);
                    H = min(C,alpha(i)+alpha(j));

                end

                if L == H
                    continue;
                end

                eta = 2*K(i,j)-K(i,i)-K(j,j);

                if eta >= 0
                    continue;
                end

                alpha(j) = alpha(j) - ...
                    Y(j)*(Ei-Ej)/eta;

                if alpha(j) > H
                    alpha(j) = H;
                elseif alpha(j) < L
                    alpha(j) = L;
                end

                if abs(alpha(j)-oldAj) < 1e-5
                    continue;
                end

                alpha(i) = alpha(i) + ...
                    Y(i)*Y(j)*(oldAj-alpha(j));

                b1 = bias - Ei ...
                    - Y(i)*(alpha(i)-oldAi)*K(i,i) ...
                    - Y(j)*(alpha(j)-oldAj)*K(i,j);

                b2 = bias - Ej ...
                    - Y(i)*(alpha(i)-oldAi)*K(i,j) ...
                    - Y(j)*(alpha(j)-oldAj)*K(j,j);

                if alpha(i) > 0 && alpha(i) < C
                    bias = b1;
                elseif alpha(j) > 0 && alpha(j) < C
                    bias = b2;
                else
                    bias = (b1+b2)/2;
                end

                changed = changed + 1;

            end

        end

        if changed == 0
            passes = passes + 1;
        else
            passes = 0;
        end

        iteration = iteration + 1;

        if mod(iteration,100) == 0
            fprintf('SVM iteration: %d\n',iteration);
        end

    end

    support = alpha > 1e-5;

    model.alpha = alpha(support);
    model.Y = Y(support);
    model.X = X(support,:);
    model.bias = bias;
    model.gamma = gamma;
    model.C = C;

end


function gamma = calculateGamma(X)

    n = size(X,1);

    distances = [];

    for i = 1:n

        for j = i+1:n

            d = X(i,:) - X(j,:);

            distances(end+1) = sum(d.^2);

        end

    end

    distances = distances(distances > 1e-12);

    if isempty(distances)

        gamma = 1;

    else

        medianDistance = median(distances);

        if medianDistance < 1e-12
            gamma = 1;
        else
            gamma = 1/(2*medianDistance);
        end

    end

end


function K = rbfKernel(X1,X2,gamma)

    n1 = size(X1,1);
    n2 = size(X2,1);

    K = zeros(n1,n2);

    for i = 1:n1

        for j = 1:n2

            difference = X1(i,:) - X2(j,:);

            distanceSquared = sum(difference.^2);

            K(i,j) = exp(-gamma*distanceSquared);

        end

    end

end


function score = svmPredict(X,model)

    K = rbfKernel(X,model.X,model.gamma);

    score = K*(model.alpha .* model.Y) + model.bias;

end