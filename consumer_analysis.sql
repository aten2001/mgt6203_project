create table consumer_analysis as (
  SELECT
    CASE WHEN cast(RIGHT(ph.rim_week, 2) AS UNSIGNED) <= 13
      THEN 1
    ELSE 0 END                                  AS Q1,
    (CASE WHEN cast(RIGHT(ph.rim_week, 2) AS UNSIGNED) > 13
               AND cast(RIGHT(ph.rim_week, 2) AS UNSIGNED) <= 26
      THEN 1
     ELSE 0 END)                                AS Q2,
    (CASE WHEN cast(RIGHT(ph.rim_week, 2) AS UNSIGNED) > 26
               AND cast(RIGHT(ph.rim_week, 2) AS UNSIGNED) <= 39
      THEN 1
     ELSE 0 END)                                AS Q3,
    (CASE WHEN cast(RIGHT(ph.rim_week, 2) AS UNSIGNED) > 39
               AND cast(RIGHT(ph.rim_week, 2) AS UNSIGNED) <= 52
      THEN 1
     ELSE 0 END)                                AS Q4,
    SUM(ph.total_units_purchased / 100)         AS total_units_purchased,
    SUM(ph.total_units_purchased_on_store_coup) AS total_units_purchased_on_mfr_coup,
    SUM(ph.total_units_purchased_on_mfr_coup)   AS total_units_purchased_on_store_coup,
    ph.rim_market,
    hd.num_large_appliances,
    hd.num_small_appliances,
    hd.num_pets,
    hd.num_members_in_household,
    hd.household_income,
    hd.male_head_avg_work_hours,
    hd.female_head_avg_work_hours,
    ph.household_id,
    ph.rim_week,
    (CASE WHEN max(hd.male_head_avg_work_hours) > max(hd.male_head_avg_work_hours)
      THEN hd.male_head_avg_work_hours
     ELSE hd.female_head_avg_work_hours END)    AS primary_head_avg_work_hours,
    ws.weekday_shopper,
    ws.weekday_purchase_percentage
  FROM purchase_history ph
    JOIN retail_tracking rt ON ph.rim_week = rt.rim_week AND ph.store_id = rt.store_id
    JOIN (SELECT
            household_id,
            (sum(washing_machine) + sum(clothes_dryer) + sum(dishwasher) + sum(freezer) + sum(convection_oven) +
             sum(trash_compactor) + sum(vacuum_cleaner) + sum(personal_computer) +
             sum(water_softner))                  AS num_large_appliances,
            (sum(toaster) + sum(toaster_broiler_oven) + sum(blender) + sum(food_processor) + sum(microwave) +
             sum(coffee_maker) + sum(garbage_disposal) + sum(hair_dryer) + sum(curling_iron) + sum(hair_rollers) +
             sum(vcr) + sum(other_appliances))    AS num_small_appliances,
            (sum(num_of_cats) + sum(num_of_dogs)) AS num_pets,
            male_head_avg_work_hours,
            female_head_avg_work_hours,
            household_income,
            num_members_in_household
          FROM household_demographics hd
          GROUP BY household_id) AS hd ON hd.household_id = ph.household_id
    JOIN weekday_shopper ws ON ws.household_id = ph.household_id
  GROUP BY hd.household_id, ph.rim_week, ph.rim_market);
