library(lars)
library(glmnet)
library(plm)

data = read.csv("~/gatech/mgt6203/project/mgt6203_project/consumer_analysis_quarterly.csv",header=TRUE,sep=",",stringsAsFactors=FALSE, row.names=NULL)
mfg_data = read.csv("~/gatech/mgt6203/project/mgt6203_project/mfg_consumer_analysis_quarterly.csv",header=TRUE,sep=",",stringsAsFactors=FALSE, row.names=NULL)

data$brand = as.factor(data$brand)
data$rim_market = as.factor(data$rim_market)
data$household_income = as.factor(data$household_income)

data$proportion_purchased_store_coup = data$total_units_purchased_on_store_coup / data$total_units_purchased


mfg_data$brand = as.factor(mfg_data$brand)
mfg_data$rim_market = as.factor(mfg_data$rim_market)
mfg_data$household_income = as.factor(mfg_data$household_income)


mfg_data$proportion_purchased_mfr_coup = mfg_data$total_units_purchased_on_mfr_coup / mfg_data$total_units_purchased


non_zero_store_data = data[data$total_units_purchased_on_store_coup != 0, ]
non_zero_mfg_data = mfg_data[mfg_data$proportion_purchased_mfr_coup > 0, ]





qplot(data$store_coup_bool, geom="histogram")
qplot(mfg_data$mfg_coup_bool, geom="histogram")

qplot(non_zero_store_data$proportion_purchased_store_coup, geom="histogram")
qplot(log(non_zero_store_data$proportion_purchased_store_coup), geom="histogram")
qplot(sqrt(non_zero_store_data$proportion_purchased_store_coup), geom="histogram") 

qplot(non_zero_mfg_data$proportion_purchased_mfr_coup, geom="histogram") 
qplot(log(non_zero_mfg_data$proportion_purchased_mfr_coup), geom="histogram")
qplot(sqrt(non_zero_mfg_data$proportion_purchased_mfr_coup), geom="histogram") 



mylogit <- glm(store_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + rim_market + num_large_appliances + 
                 num_pets + num_members_in_household + household_income + 
                 total_head_avg_work_hours + weekday_purchase_percentage + 
                 residence_status, data = data, family = "binomial")
summary(mylogit)

mylogit <- glm(mfg_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + rim_market + num_large_appliances + 
                 num_pets +  household_income + 
                 total_head_avg_work_hours + weekday_purchase_percentage, data = mfg_data, family = "binomial")
summary(mylogit)





### Gamma

#store
p_model <- glm(sqrt(proportion_purchased_store_coup) ~ Q2 + Q3 + Q4 + 
                 year + rim_market + 
                 num_members_in_household + 
                 weekday_purchase_percentage, family=Gamma(link = "log"), data=non_zero_store_data)
summary(p_model)

#manufacturer
p_model <- glm(sqrt(proportion_purchased_mfr_coup) ~ Q2 + Q3 + Q4 + 
                 year + rim_market + num_large_appliances + 
                 num_pets + num_members_in_household + 
                 total_head_avg_work_hours , family=Gamma(link = "log"), data=non_zero_mfg_data)
summary(p_model)






cluster_data = read.csv("~/gatech/mgt6203/project/mgt6203_project/consumer_analysis_quarterly_cluster_results.csv",header=TRUE,sep=",",stringsAsFactors=FALSE, row.names=NULL)
mfg_cluster_data = read.csv("~/gatech/mgt6203/project/mgt6203_project/mfg_consumer_analysis_quarterly_cluster_results.csv",header=TRUE,sep=",",stringsAsFactors=FALSE, row.names=NULL)

#cluster_$brand = as.factor(cluster_$brand)
cluster_data$rim_market = as.factor(cluster_data$rim_market)
cluster_data$household_income = as.factor(cluster_data$household_income)

cluster_data$proportion_purchased_store_coup = cluster_data$total_units_purchased_on_store_coup / cluster_data$total_units_purchased

#mfg_cluster_data$brand = as.factor(mfg_cluster_data$brand)
mfg_cluster_data$rim_market = as.factor(mfg_cluster_data$rim_market)
mfg_cluster_data$household_income = as.factor(mfg_cluster_data$household_income)

mfg_cluster_data$proportion_purchased_mfr_coup = mfg_cluster_data$total_units_purchased_on_mfr_coup / mfg_cluster_data$total_units_purchased

mfg_cluster_data$k3 = as.factor(mfg_cluster_data$k3)
cluster_data$k3 = as.factor(cluster_data$k3)
non_zero_store_cluster_data = cluster_data[cluster_data$proportion_purchased_store_coup > 0, ]
non_zero_mfg_cluster_data = mfg_cluster_data[mfg_cluster_data$proportion_purchased_mfr_coup > 0, ]


qplot(cluster_data$store_coup_bool, geom="histogram")
qplot(mfg_cluster_data$mfg_coup_bool, geom="histogram")

### logit clustering models
mylogit <- glm(store_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + rim_market + num_large_appliances + 
                 num_pets + num_members_in_household + household_income + 
                 total_head_avg_work_hours + weekday_purchase_percentage + 
                 residence_status + k3, data = cluster_data, family = "binomial")
summary(mylogit)

# model per cluster
cluster_0 = cluster_data[cluster_data$k3 == 0, ]
cluster_1 = cluster_data[cluster_data$k3 == 1, ]
cluster_2 = cluster_data[cluster_data$k3 == 2, ]

mylogit <- glm(store_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + num_large_appliances + rim_market + 
                 num_pets + num_members_in_household + household_income + 
                 total_head_avg_work_hours + weekday_purchase_percentage, data = cluster_0, family = "binomial")
summary(mylogit)

mylogit <- glm(store_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + rim_market + 
                 num_pets + num_members_in_household + 
                 total_head_avg_work_hours + weekday_purchase_percentage, data = cluster_1, family = "binomial")
summary(mylogit)

mylogit <- glm(store_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + rim_market + 
                 num_members_in_household + 
                 total_head_avg_work_hours + weekday_purchase_percentage , data = cluster_2, family = "binomial")
summary(mylogit)

# store gamma clustering
p_model <- glm(sqrt(proportion_purchased_store_coup) ~ Q2 + Q3 + Q4 + 
                 year + rim_market + 
                 num_members_in_household + 
                 weekday_purchase_percentage, family=Gamma(link = "log"), data=non_zero_store_data)
summary(p_model)











#manufacturing logit
mylogit <- glm(mfg_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + rim_market + num_large_appliances + 
                 num_pets +  household_income + 
                 total_head_avg_work_hours + weekday_purchase_percentage + k3, data = mfg_cluster_data, family = "binomial")
summary(mylogit)

mfg_cluster_0 = mfg_cluster_data[mfg_cluster_data$k3 == 0, ]
mfg_cluster_1 = mfg_cluster_data[mfg_cluster_data$k3 == 1, ]
mfg_cluster_2 = mfg_cluster_data[mfg_cluster_data$k3 == 2, ]

mylogit <- glm(mfg_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + rim_market + 
                 household_income + 
                 total_head_avg_work_hours + weekday_purchase_percentage, data = mfg_cluster_0, family = "binomial")
summary(mylogit)

mylogit <- glm(mfg_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + rim_market + num_large_appliances + 
                 household_income + 
                 total_head_avg_work_hours + weekday_purchase_percentage, data = mfg_cluster_1, family = "binomial")
summary(mylogit)

mylogit <- glm(mfg_coup_bool ~ Q2 + Q3 + Q4 + 
                 year + rim_market + 
                 household_income + 
                 total_head_avg_work_hours, data = mfg_cluster_2, family = "binomial")
summary(mylogit)
