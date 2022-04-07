import pandas as pd
import numpy as np

from path import test1_data


root_path = test1_data
med = root_path + "admissions_med.csv"
surg = root_path + "admissions_surg.csv"
img = root_path + "img.csv"

d_med = pd.read_csv(med)  # 2305
d_surg = pd.read_csv(surg)  # 1195
d_img = pd.read_csv(img)  # 5000

# 1.Create one data frame called admissions_img, consisting of all rows
# in admissions_surg and admissions_med, merged with the imaging data using ID
# (retaining all IDs from both).

name_s = list(d_surg.head())
name_m = list(d_med.head())
d_name = {name_s[i]: mn for i, mn in enumerate(name_m)}

d_surg = d_surg.rename(columns=d_name)
admissions = d_med.append(d_surg)

# merge img.csv
admissions_img = pd.merge(admissions, d_img, on='ID')

# 2. In admissions_img, create a new length_of_stay variable
# defined as discharge date and time minus admission date and time (in days).
# Calculate the mean length_of_stay for each department.
# Briefly describe how you dealt with missing data.

# many ways to handle missing data such as removing variables, filling in
# values with most frequent values, exploring correlations/similarities with
# the other values; case-dependent. Here -- use the admission time to fill
# with the discharge time.

# deal with nan value in discharge time
admissions_img.discharge_time.fillna(admissions_img.admission_time,
                                     inplace=True)

adm_dt = pd.to_datetime(admissions_img['admission_date'] + ' ' +
                        admissions_img['admission_time'])
dis_dt = pd.to_datetime(admissions_img['discharge_date'] + ' ' +
                        admissions_img['discharge_time'])

# add variable length_of_stay
admissions_img['length_of_stay'] = pd.to_timedelta(dis_dt - adm_dt).dt.days

# calculate mean for each department
departments = list(set(admissions_img['department']))
mean_dept = {d: np.nan for d in departments}
for dept in departments:
    mean_dept[dept] = np.nanmean(admissions_img['length_of_stay'].where(
        admissions_img['department'] == dept))
# print(mean_dept)

# 2a. In imaging, filter to the first performed test
# for each test_name. Save this data frame as q3_df.

d_img['img_date'] = pd.to_datetime(d_img['performed_date'] + ' ' +
                                   d_img['performed_time'])
test_names = set(d_img['test_name'])

q3_df = pd.DataFrame()
for name in test_names:
    min_img_date = np.min(d_img['img_date'].where(d_img['test_name'] == name))
    data = d_img[(d_img['img_date'] == min_img_date) & (d_img['test_name'] ==
                                                        name)]
    q3_df = pd.concat([q3_df, data])

q3_df = q3_df.reset_index()

# 2b. Transform q3_df into wide format such that each
# test_name becomes a column displaying the performed_date of that test
# Display the head of the table.
q3_df['foo'] = ['one'] * len(q3_df['performed_date'])
q3_df2 = q3_df.pivot(index='foo', values='performed_date', columns='test_name')
# print(list(q3_df2.columns))

# 3. From admissions_img, remove any rows with missing values (NA)
# in 2 or more variables and name the resulting data frame q4_df.
# Report any non-zero missing rate (%) in this data frame.
non_col_list = admissions_img.columns[admissions_img.isnull().any(axis=0).values].tolist()
total_len = len(admissions_img)
nonzero_missing_rate = {
    col: np.round(admissions_img[col].notnull().sum() / total_len * 100, 2) for col in
    non_col_list}
q4_df = admissions_img.drop(admissions_img[
                                (admissions_img[non_col_list[0]].isnull()) & (
                                    admissions_img[non_col_list[1]].isnull()) & (
                                    admissions_img[non_col_list[2]].isnull())].index)

q4_df = q4_df.drop(
    q4_df[(q4_df[non_col_list[0]].isnull()) & (q4_df[non_col_list[1]].isnull())].index)

q4_df = q4_df.drop(
    q4_df[(q4_df[non_col_list[1]].isnull()) & (q4_df[non_col_list[2]].isnull())].index)

q4_df = q4_df.drop(
    q4_df[(q4_df[non_col_list[0]].isnull()) & (q4_df[non_col_list[2]].isnull())].index)

q4_df = q4_df.append(
    {name: nonzero_missing_rate[name] if name in non_col_list else np.nan for name in
     q4_df.head().columns.tolist()},
    ignore_index=True)

# 5. In admissions_img, impute the missing age, test_name,
# and technician_name values in any way you see fit and name the resulting data
# frame q5_df. Briefly describe why you chose that method and print
# the mean of age after imputation.
q5_df = admissions_img[(admissions_img['age'].isnull())][
    ['test_name', 'technician_name']]  # 299 missing values
q5_df2 = admissions_img[(admissions_img['age'].notnull())][
    ['age', 'test_name', 'technician_name']]

test_names = list(set(admissions_img['test_name']))
tech_names = list(set(admissions_img['technician_name']))

pct_test = q5_df['test_name'].value_counts() / q5_df2['test_name'].value_counts() * 100
pct_tech = q5_df['technician_name'].value_counts() / q5_df2[
    'technician_name'].value_counts() * 100

mean_test_n_age = pd.DataFrame.from_dict(
    {tn: np.nanmean(admissions_img['age'].where(admissions_img['test_name'] == tn)) for tn
     in test_names},
    orient='index')
mean_tech_n_age = pd.DataFrame.from_dict(
    {tn: np.nanmean(admissions_img['age'].where(admissions_img['technician_name'] == tn))
     for tn in tech_names},
    orient='index')

# mean_test_n_age.sort_values(by=0).plot()
# mean_tech_n_age.sort_values(by=0).plot()


print()
