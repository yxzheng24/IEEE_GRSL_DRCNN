clear;close all;

%% settings
foldertrain = './cave_Hini/train_22Hini';
folderlabel='./cave_Hres/label_22Hres';
size_input=32;
size_label=32;
stride =16;

savepath = 'train_cave.h5';
filepaths1 = dir(fullfile(foldertrain,'*.mat'));
filepaths2 = dir(fullfile(folderlabel,'*.mat'));

%% initialization
data = zeros(size_input, size_input, 31, 1);
label = zeros(size_label, size_label, 31, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generating training data
for i=1:length(filepaths1) 
    
    imgt = load(fullfile(foldertrain,filepaths1(i).name));
    imgl = load(fullfile(folderlabel,filepaths2(i).name));
    
    image_train = imgt.I_gf;
    image_label = imgl.I_res;

    image_input=image_train;
    image_output=image_label;

    [hei,wid,c]=size(image_input);

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = image_input(x : x+size_input-1, y : y+size_input-1,:);
            subim_label = image_output(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);

            count=count+1;
            data(:, :, :, count) = subim_input;
            label(:, :, :, count) = subim_label;
        end
    end
end

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order);

%% writing to HDF5 file
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);