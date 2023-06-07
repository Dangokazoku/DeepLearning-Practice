%% P1 图像裁剪
Im = imread('image.png');
Il = imread('lable.png');
n = 0;

% 均匀裁剪图像
for i = 1:600:2401
    for j = 1:600:2401
        file_name_i = ['image\' num2str(n) '.png'];
        Ic = imcrop(Im,[i j 599 599]);
        imwrite(Ic, file_name_i)
        file_name_l = ['lable\' num2str(n) '.png'];
        Ic = imcrop(Il,[i j 599 599]);
        Ic = rgb2gray(Ic);
        imwrite(Ic, file_name_l)
        n = n+1;
    end
end

% 随机裁剪图像
for n = 25:74
    i = randi([1 2401]);
    j = randi([1 2401]);
    file_name_i = ['image\' num2str(n) '.png'];
    Ic = imcrop(Im,[i j 599 599]);
    imwrite(Ic, file_name_i)
    file_name_l = ['lable\' num2str(n) '.png'];
    Ic = imcrop(Il,[i j 599 599]);
    Ic = rgb2gray(Ic);
    imwrite(Ic, file_name_l)
end

%% P2 查看裁剪效果
CurrentFile = pwd;

imds = imageDatastore('image\');
imdl = imageDatastore('lable\');

I = readimage(imds,20);
I = histeq(I);
imshow(I);

%制作像素类别标签
classes = [
    "Buliding" 
    "Background"
           ];
pxds = pixelLabelDatastore('lable\',classes,[255,0]);

% 显示其中一张图片并叠加标注信息
C = readimage(pxds,20);
cmap = ColorMap;
B = labeloverlay(I,C,'ColorMap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap,classes);

%% P3 构建 DeeplabV3+ 网络

% 定义图像各参数
tbl = countEachLabel(pxds);
 
frequency = tbl.PixelCount/sum(tbl.PixelCount);
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

imageSize = [600 600 3];
numClasses = numel(classes);

%构建deeplabV3+.
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet50");

% 设置类别权重
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

% 修改输出层
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

%% P4 训练参数设置
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',6,...
    'LearnRateDropFactor',0.2,...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',20, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4, ...
    'ExecutionEnvironment','cpu');

% 图像增强
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);
pximds = pixelLabelImageDatastore(imds,pxds,'DataAugmentation',augmenter);

% 训练网络
[net, info] = trainNetwork(pximds,lgraph,options);

%% P5 验证
vds = imageDatastore('lable\');
I = readimage(imds,1);
C = semanticseg(I,net);
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
figure
imshow(B)
pixelLabelColorbar(cmap, classes);

%% function1：三通道色度制作标签颜色
function cmap = ColorMap()
 
cmap = [
    032 092 200   % 背景
    000 000 000   % 建筑物
        ];
 
% Normalize between [0 1].
cmap = cmap ./ 255;
end

%% fuction2：标签颜色展示
function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.
 
colormap(gca,cmap)
 
% Add colorbar to current figure.
c = colorbar('peer', gca);
 
% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);
 
% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;
 
% Remove tick mark.
c.TickLength = 0;
end
