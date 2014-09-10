clc;
clear;
close all;

samples = importdata('/tmp/distribution_test/samples.txt');
density = importdata('/tmp/distribution_test/evaluation.txt');

%%
bin_count = 100;
bin_width = (max(samples) - min(samples)) / bin_count;
approximate_density = hist(samples, bin_count) / (size(samples,1) * bin_width);


%% plot stuff
    close all

    figure;
    hold on;

    plot(density(:,1), density(:,2));
    plot(min(samples)+bin_width/2:bin_width:max(samples), approximate_density, 'r')
    
    x = 0:0.1:10;
    y = exppdf(x,1/2.5);
    plot(x,y,'g');
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

