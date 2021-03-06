

directory = '../Logos/';
ims = dir([directory '*']);
%im = imread('../images/cat.jpg');

%delete('out/*')

%CREATE TRAINING DATA FOLDERS
rmdir('training_data/*')
for i = 1: length(ims)
    if strcmp(ims(i).name(1),'.')  
        continue
    end
    
    str_parts = strsplit(ims(i).name,'.');
    im_name = str_parts(1);
    im_name = im_name{1};
    
    mkdir('training_data', im_name)
    
end

%SYNTHESIZE DATA AND POPULATE TRAINING FOLDERS
crop_param = 0.1;
for i = 1: length(ims)
    
    %ignore invisible files
    %can this be done outside main loop? ie. remove hidden files
    if strcmp(ims(i).name(1),'.')  
        continue
    end
    
    str_parts = strsplit(ims(i).name,'.');
    im_name = str_parts(1);
    im_name = im_name{1};
    
    
    im = imread([directory ims(i).name]);
    
    h = length(im(:,1,1));
    w = length(im(1,:,1));
    
    for j = 1:100

        l = round(w*crop_param*rand); r = round(w*crop_param*rand); 
        t = round(h*crop_param*rand); b = round(h*crop_param*rand);

        im_crop = im((b+1):h-t, (l+1):w-r, :);
        
        imwrite(im_crop, strcat('training_data/', im_name,...
                                '/', num2str(j), '_', ims(i).name));
                            %im_name, '_', num2str(j), '_cat.jpg'))

    end
end

%im


for i = 1:length(ims);

    %ignore invisible files
    %can this be done outside main loop? ie. remove hidden files
    if strcmp(ims(i).name(1),'.')  
        continue
    end
    
    im = imread([directory ims(i).name]);
    
    for j = 1:10
    
        im_bright = im * (1 + 0.1*(j - 5.0));
    
%        imwrite(im_bright, strcat('out/', 'bright_', num2str(j),...
%                                  '_', ims(i).name));

    end
    
%    imshow)

%im = imread('../images/cat.jpg');
end





