require(reshape2)
require(plyr)
require(ggplot2)

data <- read.csv('simulation_results_rsph6.5.csv')
data$k <- as.factor(data$k)
data <- data[,-1]
data_ <- melt(data, value.name='Value', variable.name='Index')
# relevel factor Index for plotting order
data_$Algorithm <- factor(data_$Algorithm, levels=c('complete', 'kmeans',
                                            'gmm-sph', 'gmm-diag',
                                            'gmm-tied', 'gmm-full',
                                            'ward-unstr', 'ward-str'))

data_gt <- subset(data_, Index %in% c('ARI_GT', 'AMI_GT', 'Instability_GT',
                                      'Correlation_GT'))
data_ <- subset(data_, Index %in% c('ARI', 'AMI', 'Instability', 
                                    'Correlation'))
data_gt$Index <- mapvalues(data_gt$Index, 
                           from=c('ARI_GT', 'AMI_GT', 'Instability_GT', 
                                  'Correlation_GT'),
                           to=c('ARI', 'AMI', 'Instability', 'Correlation'))

simulated_plot <- 
ggplot(data_, aes(k, Value, group=k)) + 
    geom_boxplot() + 
    # true k
    geom_vline(xintercept=6, color='blue', linetype='longdash', alpha=.8) +
    # plot ground truth curve
    stat_summary(data=data_gt, fun.y=mean, geom='line', 
                 aes(group=1), color='red') +
    stat_summary(data=data_gt, fun.y=mean, aes(group=1), color='red',
                 geom='point', shape=17, size=3) +
    # plot stability
    stat_summary(fun.y=mean, geom='line', aes(group=1)) +
    stat_summary(fun.y=mean, aes(group=1),  
                 geom='point', shape=17, size=3) +
    facet_grid(Algorithm ~ Index) + 
    theme_bw(base_size=18)

fnout <- 'simulation_results_rsph6.5.pdf'
cat(paste('Saving', fnout, '\n'))

pdf(file=fnout, onefile=F, title='', 
paper='special', width=12, height=24, bg='white')
simulated_plot
dev.off()
