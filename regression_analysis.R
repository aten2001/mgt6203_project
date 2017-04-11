data = read.csv("~/gatech/mgt6203/project/new_data.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)

data$total_units_purchased = data$total_units_purchased / 100
data$response = data$total_units_purchased_on_store_coup / data$total_units_purchased

data$male_head_race = as.factor(data$male_head_race)
data$female_head_race = as.factor(data$female_head_race)

model <- lm(formula = response~rim_market + household_income + num_members_in_household + male_head_race + female_head_race, data=data)
summary(model)

library(ggplot2)

ggplot(data=data, aes(factor(rim_market), response)) +
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


# appliance data analysis
appliance_data = read.csv("~/gatech/mgt6203/project/appliance_data.csv",header=TRUE,sep=",",stringsAsFactors=FALSE)

appliance_data$num_large_appliances = as.numeric(appliance_data$num_large_appliances)
appliance_data$num_small_appliances = as.numeric(appliance_data$num_small_appliances)

# create the response variable
appliance_data$total_units_purchased = appliance_data$total_units_purchased / 100
appliance_data$response = appliance_data$total_units_purchased_on_store_coup / appliance_data$total_units_purchased

model <- lm(formula = response~rim_market + num_large_appliances + num_small_appliances, data=appliance_data)
summary(model)

# scatterplot
ggplot(appliance_data, aes(x=num_large_appliances, y=response)) + geom_point()

# boxplots
ggplot(appliance_data, aes(x=factor(num_large_appliances), y=response)) + geom_boxplot()
ggplot(appliance_data, aes(x=factor(num_small_appliances), y=response)) + geom_boxplot()

