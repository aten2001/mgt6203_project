library(lars)
library(glmnet)
library(plm)

data = read.csv("~/gatech/mgt6203/project/mgt6203_project/consumer_analysis.csv",header=TRUE,sep=",",stringsAsFactors=FALSE, row.names=NULL)

data$proportion_purchased_store_coup = data$total_units_purchased_on_store_coup / data$total_units_purchased
data$brand = as.factor(data$brand)
data$rim_market = as.factor(data$rim_market)
data$household_income = as.factor(data$household_income)

model <- lm(formula = proportion_purchased_store_coup ~ Q2 + Q3 + Q4 + rim_market +  
                      num_small_appliances + num_members_in_household + household_income + 
                      primary_head_avg_work_hours, data=data)
summary(model)


data$proportion_purchased_mfr_coup = data$total_units_purchased_on_mfr_coup / data$total_units_purchased
model <- lm(formula = proportion_purchased_mfr_coup ~ Q2 + Q3 + Q4 + rim_market + num_large_appliances + 
              num_small_appliances + num_members_in_household + household_income + 
              primary_head_avg_work_hours, data=data)
summary(model)

# https://stats.stackexchange.com/questions/66973/difference-between-fixed-effects-models-in-r-plm-and-stata-xtreg

fe_model <- plm(proportion_purchased_store_coup ~ Q2 + Q3 + Q4 + rim_market + num_large_appliances + 
                  num_small_appliances + num_pets + num_members_in_household + household_income + 
                  primary_head_avg_work_hours, 
               data   = data, 
               method = "within", #fixed effects model
               effect = "time", #does the gamma_i and delta_t parts (I think)
               index  = c("brand")
              )

summary(fe_model)

library(ggplot2)

ggplot(data=data, aes(factor(rim_market), proportion_purchased_store_coup)) +
  geom_boxplot() + 
  xlab("ERIM Market") + 
  ylab("Units purchased w coupon/units purchased")

ggplot(data=data, aes(factor(household_income), response)) +
  geom_boxplot() + 
  xlab("Household Income") + 
  ylab("Units purchased w coupon/units purchased")

ggplot(data=data, aes(factor(num_members_in_household), response)) +
  geom_boxplot() + 
  xlab("Number of members in household") + 
  ylab("Units purchased w coupon/units purchased")


ggplot(data=data, aes(factor(male_head_race), response)) +
  geom_boxplot() + 
  xlab("Male head race") + 
  ylab("Units purchased w coupon/units purchased")


ggplot(data=data, aes(factor(female_head_race), response)) +
  geom_boxplot() + 
  xlab("Female head race") + 
  ylab("Units purchased w coupon/units purchased")


