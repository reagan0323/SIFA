function img = MakeImageRB(cor)

% positive is red, negative is blue, zero is white
    [m n] = size(cor);

	palUp = uint8(0:255)';
	palDw = uint8(255:-1:0)';
%   palZr = zeros(256,1,'unit8');
    palZr = zeros(256,1,'uint8')+255;
%	palIm = [palZr palDw palZr; palUp palZr palZr];
    palIm=[palUp palUp palZr ; palZr palDw palDw];


    npal = size(palIm,1);
    palCenter = (npal+1)/2;

    level = floor(sqrt(numel(cor)));
	sorted = sort(cor(:)); % sort all elements in cor and output a vector
    subset = abs([sorted(1:level); sorted(end-level+1:end)]); % select the absolute large values
    clear sorted;
    sorted2 = sort(subset,'descend'); % sort selected elements
    maxDist = sorted2(level); % middle one of large values    
    clear level subset sorted2;
    
% 	maxDist = sorted(floor(0.999*end));
%     clear sorted;
% 	maxDist = max(stdist(floor(0.99*end)),  stdist(ceil(0.01*end)));
% 	minDist = -maxDist;

	imageInd = cor/maxDist*(npal/2) + palCenter;
    imageInd = round(imageInd);
	imageInd(imageInd<1) = 1;
	imageInd(imageInd>npal) = npal;
    
	img = reshape(  palIm(imageInd,:) ,[m,n,3]);
    clear imageInd;
	image(img)
return;