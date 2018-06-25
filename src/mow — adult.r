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
    is_any = FALSE
    for(id in 1:nrow(data)){
      data[id, paste0(name,'_',as.character(data[id,name]))] = 1
    } 
  }
  data[,!(names(data) %in% columns)]
}


# Adult data

adult = read.csv('adult.data', header = TRUE)
set.seed(100)
indexes = sample(nrow(adult),size=12000,replace=FALSE)
adult = adult[indexes,]
catigorical_columns = c('workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country')
adult_clusters = as.numeric(adult$class) - 1
adult$class = NULL
adult = categorical_to_numerical(adult, catigorical_columns)
adult$fnlwgt = NULL
adult = adult[, !(colSums(adult == 0) == nrow(adult))]
adult = scale(adult)

table(is.na(adult))

table(adult_clusters)

# Analiza

data = adult
clusters = adult_clusters

# K - œrednich

internal_index = NULL
external_index = NULL

for(n in 1:15){
  cat('K - œrednich', n, '\n')
  group_kmeans = kmeans(data, n, iter.max = 30)

  internal_index = rbind(internal_index, intCriteria(as.matrix(data), group_kmeans$cluster, c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(group_kmeans$cluster, as.integer(clusters), c('Rand','Czekanowski_Dice')))
}
k_means_index = data.frame(internal_index, external_index)

# DBSCAN


dbscan_k = 6
eps_param = c(15, 10, 5, 5, 4, 2)
min_pts_param = c(20, 40, 80, 20, 40, 20)

internal_index = NULL
external_index = NULL
dbscan_index = NULL
groups_count = NULL
group_0_count = NULL
group_1_count = NULL
for(i in 1:dbscan_k){
  cat('DBSCAN', i, '\n')
  group_dbscan = dbscan(data, eps = eps_param[i], MinPts = min_pts_param[i])

  internal_index = rbind(internal_index, intCriteria(as.matrix(data), as.integer(group_dbscan$cluster), c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(as.integer(group_dbscan$cluster), as.integer(clusters), c('Rand','Czekanowski_Dice')))
  groups_count = c(groups_count, max(group_dbscan$cluster))
  group_0_count = c(group_0_count, table(group_dbscan$cluster)[[1]])
  group_1_count = c(group_1_count, table(group_dbscan$cluster)[[2]])
}
dbscan_index = data.frame(internal_index, external_index, groups_count, group_0_count, group_1_count)


# Najdalsze s¹siedztwo

internal_index = NULL
external_index = NULL
h_grouping = hclust(dist(data), method = "complete")

h_grouping_2 <- as.dendrogram(h_grouping)
plot(h_grouping_2)
plot(cut(h_grouping_2, h=30)$upper)

plot(h_grouping)

for(n in 1:15){
  cat('Najdalsze s¹siedztwo', n, '\n')
  
  group_complete_linkage = cutree(h_grouping, k = n)
  
  internal_index = rbind(internal_index, intCriteria(as.matrix(data), group_complete_linkage, c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(group_complete_linkage, as.integer(clusters), c('Rand','Czekanowski_Dice')))
}
complete_linkage_index = data.frame(internal_index, external_index)

# SOM (Neuron jako grupa)

som_1_k = 8
xdim_1_param  = c(1, 1, 1, 2, 2, 2, 3, 3)
ydim_1_param = c(2, 3, 4, 2, 3, 4, 3, 4)
rlen_1_param = 100

internal_index = NULL
external_index = NULL
group_counts_index = NULL

for(i in 1:som_1_k)
{
  cat('SOM (Neuron jako grupa)', i, '\n')
  som_model = som(as.matrix(scale(data)), 
                  grid=somgrid(xdim = xdim_1_param[i], ydim = ydim_1_param[i], topo="hexagonal"), 
                  rlen=rlen_1_param,
                  n.hood='circular')


  internal_index = rbind(internal_index, intCriteria(as.matrix(data), som_model$unit.classif, c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(som_model$unit.classif, as.integer(clusters), c('Rand','Czekanowski_Dice')))
  group_counts_index = rbind(group_counts_index, list(table(som_model$unit.classif)))
  
}

som_1_index = data.frame(internal_index, external_index, group_counts_index)

# SOM + najdalsze s¹siedztwo

n = c(rep(2,3), rep(3,3),rep(4,3),rep(5,3),rep(8,3))
som_2_k = 5*3
xdim_2_param  = rep(c(15, 25, 30), 5)
ydim_2_param = rep(c(60, 25, 30), 5)
rlen_2_param = 10

internal_index = NULL
external_index = NULL
group_counts_index = NULL
for(i in 1:som_2_k)
{
  cat('SOM + najdalsze s¹siedztwo', i, ' ')
  som_model = som(data, 
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
  group_counts_index = rbind(group_counts_index, list(table(som_cluster)))
}

som_2_index = data.frame(internal_index, external_index, group_counts_index)

# SOM + k-œrednich

internal_index = NULL
external_index = NULL
group_counts_index = NULL
for(i in 1:som_2_k)
{
  cat('SOM + k-œrednich', i, ' ')
  som_model = som(data, 
                  grid=somgrid(xdim = xdim_2_param[i], ydim = ydim_2_param[i], topo="hexagonal"), 
                  rlen=rlen_2_param,
                  n.hood='circular')
  som_cluster_tmp = kmeans(som_model$codes, n[i])$cluster
  som_cluster = som_model$unit.classif
  for(j in 1:length(som_cluster_tmp)){
    som_cluster[som_model$unit.classif==j] = som_cluster_tmp[[j]]
  }

  internal_index = rbind(internal_index, intCriteria(as.matrix(data), som_cluster, c('Davies_Bouldin','Dunn')))
  external_index = rbind(external_index, extCriteria(som_cluster, as.integer(clusters), c('Rand','Czekanowski_Dice')))
  group_counts_index = rbind(group_counts_index, list(table(som_cluster)))
}

som_3_index = data.frame(internal_index, external_index, group_counts_index)



write.table(as.matrix(k_means_index), 'adult_kmeans.csv')
write.table(as.matrix(dbscan_index), 'adult_dbscan.csv')
write.table(as.matrix(complete_linkage_index), 'adult_cl.csv')
write.table(as.matrix(som_1_index), 'adult_som1.csv')
write.table(as.matrix(som_2_index), 'adult_som2.csv')
write.table(as.matrix(som_2_index), 'adult_som3.csv')