clear;close all;

%% settings
addpath('ST_test');
folder = './data_process/pavia_subimgs';
ratio = 4;
overlap = 1:71;
size_kernel=[8 8];
sig = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1; 
start_pos(2)=1;
%% Parameters for structure tensor analysis and statistics
sigma_DoG = 0.27;  % Standard deviation of derivative-of-gaussian (DoG) kernels [pixel]
sigma_gauss = 0.5;   % Standard deviation of Gaussian kernel [pixel]
%% Perform ST analysis and directional statistics
% Create Structure Tensor struct
ST = CreateStructureTensorStruct(sigma_gauss,sigma_DoG,true);

hl = fspecial('log',[15 15],0.43);
hg = fspecial('gaussian',[25 25],0.43);
count = 0;
filepaths = dir(fullfile(folder,'*.mat'));

for i = 1 : length(filepaths)
    
    image_s=load(fullfile(folder,filepaths(i).name));
    image = image_s.subim;
    %% Normalization 
    Amax=max(max(max(image)));
    Bmin=min(min(min(image)));
    I_REF=(image-Bmin)./(Amax-Bmin);
    %% Obtaining the spatial details of the enhanced PAN image via ST
    PAN = mean(I_REF(:,:,overlap),3);
    D_PAN = PAN - imfilter(PAN,hl,'replicate');
    ST = ApplyStructureTensorStruct(D_PAN,ST);
    S_pan = D_PAN .* (ST.Tensor.Trace > 1e-5);
    
    [HSI,KerBlu]=conv_downsample(I_REF,ratio,size_kernel,sig,start_pos);
    I_HS = interp23tapGeneral(HSI,ratio);
    S_res = I_REF - I_HS;
%% Extracting the spatial information of the upsampled HSI using GF
for ii =  1 : size(I_HS,3)
    G = I_HS(:,:,ii);
    B(:,:,ii) = imfilter(G,hg,'circular');  
    B1= B(:,:,ii); 
    S_H1(:,:,ii) = imguidedfilter(B1,G,'NeighborhoodSize',15,'DegreeOfSmoothing',0.001^2); 
    S_panlhs(:,:,ii) = S_H1(:,:,ii) - B1 + S_pan;
end

fname = strcat('SpSl_',filepaths(i).name);
save(['./Pavia_SpanSlhs/',fname],'S_panlhs');

fname1 = strcat('Hu_',filepaths(i).name);
save(['./Pavia_Hu/',fname1],'I_HS');

fname2 = strcat('Sres_',filepaths(i).name);
save(['./Pavia_Sres/',fname2],'S_res');
count=count+1;
end
