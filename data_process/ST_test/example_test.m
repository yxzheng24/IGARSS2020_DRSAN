
% Parameters for structure tensor analysis and statistics
sigma_DoG = 0.27;  % Standard deviation of derivative-of-gaussian (DoG) kernels [pixel]
sigma_gauss = 0.5;   % Standard deviation of Gaussian kernel [pixel]

%% Perform ST analysis and directional statistics

% Create Structure Tensor struct
ST = CreateStructureTensorStruct(sigma_gauss,sigma_DoG,true);

h = fspecial('log',[15 15],0.43);  
B_PAN = imfilter(I_pan,h,'replicate'); 
D_PAN = I_pan-B_PAN; 
% Calculate the Structure Tensor (an additional input for multi-thread
% processing can be added)
ST = ApplyStructureTensorStruct(D_PAN,ST);
S_pan = D_PAN .* (ST.Tensor.Trace > 1e-5);