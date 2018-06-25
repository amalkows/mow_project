############################################################# 
#############################################################
############                 MOW                 ############
############    Artur Brodzki, Adam Ma³kowski    ############
############################################################# 
############################################################# 

# install.packages('kohonen')
# install.packages('som')
# install.packages('clusterCrit')

library(clusterCrit)
library(fpc)
library(caret)
library(kohonen)


categorical_to_numerical = function(data, columns){
  for(name in columns)
  {
    attsValues = levels(data[[name]]);
    for(value in attsValues){    
      data[paste0(name,'_',value)] = 0;
    } 
    
    for(id in 1:nrow(data)){
      data[id, paste0(name,'_',as.character(data[id,name]))] = 1
    } 
  }
  data[,!(names(data) %in% columns)]
}


# Car evaluation

car = read.csv('car.data', header = TRUE)
car_without_preprocessing = car
car_clusters = car$class
car$class = NULL
car = categorical_to_numerical(car, names(car))

# Analiza

data = car
clusters = car_clusters

# K - œrednich

internal_index = NULL
external_index = NULL
for(n in 1:10){
  group_kmeans = kmeans(data, n, iter.max = 30)
  png(paste('ksrednich_wykres_', n, '.png', sep=""), width = 1000, height = 1000)
  plot(car_without_preprocessing, col = group_kmeans$cluster)
  dev.off()
  internal_index = rbind(internal_index, intCriteria(as.matrix(data), group_kmeans$cluster, c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(group_kmeans$cluster, as.integer(clusters), c('Rand','Czekanowski_Dice')))
}
k_means_index = data.frame(internal_index, external_index)


# DBSCAN

dbscan_k = 6
eps_param = c(sqrt(2),sqrt(2),sqrt(2),sqrt(4),sqrt(6),sqrt(8))
min_pts_param = c(10, 30, 150, 150, 250, 500)

internal_index = NULL
external_index = NULL
dbscan_index = NULL
groups_count = NULL

for(i in 1:dbscan_k){
  group_dbscan = dbscan(data, eps = eps_param[i], MinPts = min_pts_param[i])
  group_dbscan
  group_dbscan$cluster

  internal_index = rbind(internal_index, intCriteria(as.matrix(data), as.integer(group_dbscan$cluster), c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(as.integer(group_dbscan$cluster), as.integer(clusters), c('Rand','Czekanowski_Dice')))
  groups_count = c(groups_count, max(group_dbscan$cluster))
}
dbscan_index = data.frame(internal_index, external_index, groups_count)

# Najdalsze s¹siedztwo

internal_index = NULL
external_index = NULL
h_grouping = hclust(dist(data), method = "complete")

for(n in 1:10){
  group_complete_linkage = cutree(h_grouping, k = n)
  
  internal_index = rbind(internal_index, intCriteria(as.matrix(data), group_complete_linkage, c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(group_complete_linkage, as.integer(clusters), c('Rand','Czekanowski_Dice')))
}
complete_linkage_index = data.frame(internal_index, external_index)

# SOM (Neuron jako grupa)

som_1_k = 8
som_2_k = 1
xdim_1_param  = c(1, 1, 1, 2, 2, 2, 3, 3)
ydim_1_param = c(2, 3, 4, 2, 3, 4, 3, 4)
rlen_1_param = 100

internal_index = NULL
external_index = NULL

for(i in 1:som_1_k)
{
  som_model = som(as.matrix(scale(data)), 
                  grid=somgrid(xdim = xdim_1_param[i], ydim = ydim_1_param[i], topo="hexagonal"), 
                  rlen=rlen_1_param,
                  n.hood='circular')

  internal_index = rbind(internal_index, intCriteria(as.matrix(data), som_model$unit.classif, c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(som_model$unit.classif, as.integer(clusters), c('Rand','Czekanowski_Dice')))
  
}

som_1_index = data.frame(internal_index, external_index)

# SOM + najdalsze s¹siedztwo

n = c(rep(2,3), rep(3,3),rep(4,3),rep(5,3),rep(8,3))
som_2_k = 5*3
xdim_2_param  = rep(c(10, 15, 20), 5)
ydim_2_param = rep(c(40, 15, 20), 5)
rlen_2_param = 100

internal_index = NULL
external_index = NULL

for(i in 1:som_2_k)
{
  som_model = som(as.matrix(scale(data)), 
                  grid=somgrid(xdim = xdim_2_param[i], ydim = ydim_2_param[i], topo="hexagonal"), 
                  rlen=rlen_2_param,
                  n.hood='circular')
  
  som_cluster_tmp = cutree(hclust(dist(som_model$codes)), n[i])
  som_cluster = som_model$unit.classif
  for(j in 1:length(som_cluster_tmp)){
    som_cluster[som_model$unit.classif==j] = som_cluster_tmp[[j]]
  }
  
  internal_index = rbind(internal_index, intCriteria(as.matrix(data), som_cluster, c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(som_cluster, as.integer(clusters), c('Rand','Czekanowski_Dice')))
  
}

som_2_index = data.frame(internal_index, external_index)