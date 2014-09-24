clc;
clear;
close all;

samples = importdata('/tmp/distribution_test/samples.txt');
density = importdata('/tmp/distribution_test/evaluation.txt');

%%
bin_count = 1000;
bin_width = (max(samples) - min(samples)) / bin_count;
approximate_density = hist(samples, bin_count) / (size(samples,1) * bin_width);


%% plot stuff
    close all

    figure;
    hold on;

    plot(density(:,1), density(:,2),'b');
    plot(min(samples)+bin_width/2:bin_width:max(samples), approximate_density, 'r.')
    
%      x = -3:0.01:9;
%      y = normpdf(x,3,1);
%      plot(x,y,'g');
     
%      axis([3.460380000000000 3.461920000000000 0 0.002]);
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

