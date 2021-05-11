clear;close all;

%% Load dataset
folder1 = './cave_ref';
folder2 = './cave_hspan';
count = 0;

filepaths1 = dir(fullfile(folder1,'*.mat'));
filepaths2 = dir(fullfile(folder2,'*.mat'));

 for i = 1 : length(filepaths1)
    
    image_ref=load(fullfile(folder1,filepaths1(i).name));
    I_REF =  image_ref.HR;
    [h,w,c] = size(I_REF);
     
    image_cave=load(fullfile(folder2,filepaths2(i).name));
    I_HS =  image_cave.HSI;
    I_PAN =  image_cave.PAN;
    
%% Enhancing Edge Details of the PAN Image With CLAHE 
D_PAN = adapthisteq(I_PAN);
     
I_HS2 = imresize(I_HS,[h w],'bicubic');

%% Generating the Initialized HSI via Guided Filter
for j =  1 : c
    I_gf(:,:,j) = imguidedfilter(I_HS2(:,:,j),D_PAN,'NeighborhoodSize',15,'DegreeOfSmoothing',0.001^2); 
end    

I_res = I_REF - I_gf;

fname = strcat('cave_Hini_',filepaths1(i).name);
save(['./cave_Hini/',fname],'I_gf');

fname2 = strcat('cave_Hres_',filepaths1(i).name);
save(['./cave_Hres/',fname2],'I_res');

count=count+1;
 end