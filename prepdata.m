% import fil list and class labels
close;clear;clc;
project_name = 'phpmyadmin';
pathprefix = '/Users/sbicer/Desktop/akademik/tensorflow';
filename = strcat(pathprefix,'/',project_name,'-files.csv');
delimiter = ',';
startRow = 2;

folder_name = strcat(project_name,'-data');

%   column1: text (%s)
%	column2: text (%s)
formatSpec = '%s%s%[^\n\r]';

fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);

Name = dataArray{:, 1};
Defected = dataArray{:, 2};

clearvars filename delimiter startRow formatSpec fileID dataArray ans;

celldata = cell(length(Name),1);
for i=1:length(Name)
    celldata(i,1) = regexprep(cellstr(fileread(fullfile(pathprefix,'data',Name{i}))),'\r\n|\n|\r','');
end

ip=1; posdata=cell(0);
in=1; negdata=cell(0);

for i=1:length(Defected)
    if (strcmp(Defected{i},'yes'))
        negdata(in,1)=celldata(i,1);
        in=in+1;
    else
        posdata(ip,1)=celldata(i,1);
        ip=ip+1;
    end
end

f=fopen(fullfile(pathprefix,'data',folder_name,'sourcecodes.neg'),'w');
[nrows,~] = size(negdata);
for i = 1:nrows
    fprintf(f,'%s\n',negdata{i,:});
end
fclose(f);

f=fopen(fullfile(pathprefix,'data',folder_name,'sourcecodes.pos'),'w');
[nrows,~] = size(posdata);
for i = 1:nrows
    fprintf(f,'%s\n',posdata{i,:});
end
fclose(f);

%create 10 independent validation sets.
for i=1:10
    %negative samples
    [n,~] = size(negdata);
    negdata=negdata(randperm(length(negdata)));
    f=fopen(fullfile(pathprefix,'data',folder_name,strcat('sourcecodes-',int2str(i),'.neg')),'w');
    cutoff = idivide(int32(n*7),10);
    for j=1:cutoff
        fprintf(f,'%s\n',negdata{j,:});
    end
    fclose(f);

    f=fopen(fullfile(pathprefix,'data',folder_name,strcat('sourcecodes-',int2str(i),'-val.neg')),'w');
    cutoff = cutoff+1;
    for j=cutoff:n
        fprintf(f,'%s\n',negdata{j,:});
    end
    fclose(f);

    %positive samples
    [n,~] = size(posdata);
    posdata=posdata(randperm(length(posdata)));
    f=fopen(fullfile(pathprefix,'data',folder_name,strcat('sourcecodes-',int2str(i),'.pos')),'w');
    cutoff = idivide(int32(n*7),10);
    for j=1:cutoff
        fprintf(f,'%s\n',posdata{j,:});
    end
    fclose(f);

    f=fopen(fullfile(pathprefix,'data',folder_name,strcat('sourcecodes-',int2str(i),'-val.pos')),'w');
    cutoff = cutoff+1;
    for j=cutoff:n
        fprintf(f,'%s\n',posdata{j,:});
    end
    fclose(f);
end

clearvars n f i j in ip nrows pathprefix celldata ans;
