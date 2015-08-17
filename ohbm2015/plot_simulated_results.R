require(reshape2)
require(plyr)
require(ggplot2)

fnin <- commandArgs(TRUE)[1]
data <- read.csv(fnin)
names(data)[1] <- 'k'
data$k <- as.factor(data$k)
data_ <- melt(data, value.name='Score', variable.name='Metric', 
              id.vars=c('k', 'method'))
# relevel factors for plotting order
data_$Algorithm <- factor(data_$method, levels=c('complete', 'kmeans',
                                            'gmm-sph', 'gmm-diag',
                                            'gmm-tied', 'gmm-full',
                                            'ward-unstr', 'ward-str'))
data_$Metric <- factor(data_$Metric, levels=
                           c('ARI_gt', 'AMI_gt', 'InstabilityScore_gt', 
                             'CorrelationScore_gt','ARI', 'AMI', 
                             'InstabilityScore', 'CorrelationScore'))

data_gt <- subset(data_, Metric %in% c('ARI_gt', 'AMI_gt', 'InstabilityScore_gt',
                                      'CorrelationScore_gt'))
data_ <- subset(data_, Metric %in% c('ARI', 'AMI', 'InstabilityScore', 
                                    'CorrelationScore'))
data_gt$Metric <- mapvalues(data_gt$Metric, 
                           from=c('ARI_gt', 'AMI_gt', 'InstabilityScore_gt', 
                                  'CorrelationScore_gt'),
                           to=c('ARI', 'AMI', 'InstabilityScore', 
                                'CorrelationScore'))

simulated_plot <- 
ggplot(data_, aes(k, Score, group=k)) + 
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
    facet_grid(Algorithm ~ Metric) + 
    theme_bw(base_size=18)

fnout <- sub('.csv', '.pdf', fnin)
cat(paste('Saving', fnout, '\n'))

pdf(file=fnout, onefile=F, title='', 
paper='special', width=12, height=24, bg='white')
simulated_plot
dev.off()
