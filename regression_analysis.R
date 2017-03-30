data = read.csv("~/gatech/mgt6203/project/total_num_coup_units.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)

model <- lm(formula = total_num_coup_units~., data=data)
summary(model)

library(ggplot2)

ggplot(data=data, aes(factor(rim_market), total_num_coup_units)) +
  geom_boxplot() + 
  xlab("ERIM Market") + 
  ylab("Total # coupon units")

ggplot(data=data, aes(factor(household_income), total_num_coup_units)) +
  geom_boxplot() + 
  xlab("Household Income") + 
  ylab("Total # coupon units")

ggplot(data=data, aes(factor(num_members_in_household), total_num_coup_units)) +
  geom_boxplot() + 
  xlab("Number of members in household") + 
  ylab("Total # coupon units")

ggplot(data=data, aes(factor(male_head_race), total_num_coup_units)) +
  geom_boxplot() + 
  xlab("Male head race") + 
  ylab("Total # coupon units")

ggplot(data=data, aes(factor(female_head_race), total_num_coup_units)) +
  geom_boxplot() + 
  xlab("Female head race") + 
  ylab("Total # coupon units")
