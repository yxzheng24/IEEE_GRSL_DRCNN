clear;close all;

%% Load dataset
folder = './cave_ref';

%% Settings
ratio = 8;
overlap = 1:31;
size_kernel=[8 8];
sig = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1; 
start_pos(2)=1;
count = 0;

filepaths = dir(fullfile(folder,'*.mat'));

%% Genertate LR-HSI, HR-PAN
 for i = 1 : length(filepaths)
    
image_ref = load(fullfile(folder,filepaths(i).name));
I_REF =  image_ref.ori_ini;   
    
[HSI,KerBlu]=conv_downsample(I_REF,ratio,size_kernel,sig,start_pos);
PAN = mean(I_REF(:,:,overlap),3);

fname = strcat('HSPAN_',filepaths(i).name);
save(['./cave_hspan/',fname],'HSI','PAN');
count=count+1;
 end