import zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.max_rows = 5
nobels = pd.read_csv('nobels.csv', names=['year', 'discipline', 'nobelist'])

#TODO 'nobels.csv'

nobels.tail()
nobels.head()
len(nobels)
nobels.columns
nobels.dtypes
nobels.index
nobels['discipline'], nobels.discipline
# OR
nobels.nobelist, nobels['nobelist']

nobels.discipline.values[:50]
nobels.discipline.unique()
nobels.nobelist.value_counts()

# select rows by building a Boolean mask, and using it as fancy index
nobels[nobels.discipline == 'Physics']

# select rows by the string-expression-based query interface
nobels.query('discipline == "Chemistry"')
# use single quotes for the query and double quotes for any values inside it.

nobels['Curie' in nobels.nobelist]  # not work

# Series.str methods perform operations on Series values as strings;
# "contains" tests if a pattern is contained in each element;
# so we select the rows in which the field "nobelist" contains "Curie"
nobels[nobels.nobelist.str.contains('Curie')]

disco = np.load('discography.npy')
disco_df = pd.DataFrame(disco)
disco_df.dtypes
pd.DataFrame([{'title': 'David Bowie', 'year': 1969},
              {'title': 'The Man Who Sold the World', 'year': 1970},
              {'title': 'Hunky Dory', 'year': 1971}])

pd.DataFrame([('Ziggy Stardust', 1), ('Aladdin Sane', 1), ('Pin Ups', 1)],
             columns=['title', 'toprank'])
disco['title'], disco['release']

##indexing
nobels.index
nobels_by_year = nobels.set_index('year')
nobels_by_year.index
# using loc
nobels_by_year.loc[1901]
nobels_by_year.loc[1901, 'nobelist']  # 2nd argument is column name
# slicing
nobels_by_year.loc[1914:1918]
nobels_by_discipline = nobels.set_index('discipline').sort_index()
nobels_by_discipline.head()
nobels_by_discipline.loc['Physics']
nobels_by_discipline.loc['Medicine':'Peace']

# numpy styled indexing
nobels_by_year.iloc[0:10]

# multi-indexing
nobels_multi = nobels.set_index(['year', 'discipline'])
nobels_multi.index.get_level_values(0)  # all years
nobels_multi.index.get_level_values(1)  # all disciplines
nobels_multi.loc[(2017, 'Physics')]

# nobels_multi.loc[(1901:1910, 'Chemistry')] not work
nobels_multi.loc[(slice(1901, 1910), 'Chemistry')]  # not work

# to avoid multi-indexing ambiguity, we specify a range of columns (here ":" for all of them)
nobels_multi.loc[(slice(1901, 1910), 'Chemistry'), :]  # WORK!!

# slice(None) is longhand for the end-to-end slice ":"
nobels_multi.loc[(slice(None), ['Chemistry', 'Physics']), :]

# make three Boolean masks based on year and discipline values;
# combine them element-by-element with logical AND; use result as fancy index
nobels[(nobels.year >= 1901) & (nobels.year <= 1910) & (
        nobels.discipline == 'Chemistry')]

nobels.query('year >= 1901 and year <= 1910 and discipline == "Chemistry"')

# PLOT
# compare mutiple plots, the axe of 1st plot needs to pass to the 2nd plot
gapminder = pd.read_csv('gapminder.csv') #TODO
gapminder.describe()
# create a new Series by doing numpy math on a DataFrame column;
# use dict-like syntax to assign the new Series to a new column in the DataFrame
gapminder['log_gdp_per_day'] = np.log10(gapminder['gdp_per_capita'] / 365.25)
gapminder_by_year = gapminder.set_index('year').sort_index()
gapminder_by_country = gapminder.set_index('country').sort_index()

# to superimpose multiple Pandas plots, save the axes object returned by the first,
# pass it as "ax" to further plots
axes = gapminder_by_year.loc[1960].plot.scatter('log_gdp_per_day',
                                                'life_expectancy', label=1960)
gapminder_by_year.loc[2015].plot.scatter('log_gdp_per_day', 'life_expectancy',
                                         label=2015, color='C1', ax=axes)

axes = gapminder_by_country.loc['Italy'].sort_values('year').plot('year',
                                                                  'life_expectancy',
                                                                  label='Italy')
gapminder_by_country.loc['China'].sort_values('year').plot('year',
                                                           'life_expectancy',
                                                           label='China',
                                                           ax=axes)
gapminder_by_country.loc['United States'].sort_values('year').plot('year',
                                                                   'life_expectancy',
                                                                   label='USA',
                                                                   ax=axes)

plt.axis(xmin=1900)
plt.ylabel('life expectancy')

# compute all-countries mean of babies_per_woman after segmenting data by year;
# result is Series indexed by year
gapminder.groupby('year').babies_per_woman.mean().plot(
    ylabel='babies per woman')
# with secondary_y = True, the second plot generate a second set of axis labels
gapminder.groupby('year').age5_surviving.mean().plot(secondary_y=True,
                                                     ylabel='age 5 survival [%]')

# pivot table: segment babies_per_woman data by both year and region, then take mean
gapminder.pivot_table('babies_per_woman', 'year',
                      'region')  # pivot_table is default by taking mean, but can be changed.
plt.title('babies per woman')

# Load multiple text file using Pandas
zipfile.ZipFile('names.zip').extractall('..') #TODO
open('names/yob2011.txt', 'r').readlines()[:10]
pd.read_csv('names/yob2011.txt')
pd.read_csv('names/yob2011.txt', names=['name', 'sex', 'number'])

# load CSV file as DataFrame, then create a new column "year" with all elements set to 2011
pd.read_csv('names/yob2011.txt', names=['name', 'sex', 'number']).assign(
    year=2011)

# for each year in 1880-2018, load the corresponding CSV file names/yobXXXX.txt
# as DataFrame, create new column "year" with all elements set to loop variable,
# then concatenate all DataFrames into a single one
allyears = pd.concat(pd.read_csv(f'names/yob{year}.txt',
                                 names=['name', 'sex', 'number']).assign(
    year=year)
                     for year in range(1880, 2019))

allyears.info()
allyears.year.min(), allyears.year.max()
# save DataFrame to compressed CSV file, droplting uninteresting index
allyears.to_csv('allyears.csv.gz', index=False)

# baby born, sex, name, year
allyears = pd.read_csv('allyears.csv.gz')
allyears_indexed = allyears.set_index(['sex', 'name', 'year']).sort_index()
# normalize F/Mary time series by the total number of births each year
plt.plot(allyears_indexed.loc[('F', 'Mary')] / allyears.groupby(
    'year').sum())  # This reveal the # of name Mary changes over years.

# unstack
# "pivot" the third level (level=2) of the multi-index (years) ['sex','name','year'] to create a row of columns;
# result is names (rows) x years (columns)
claires = ['Claire', 'Clare', 'Clara', 'Chiara', 'Ciara']
allyears_indexed.loc[('F', claires), :].unstack(level=2)
# OR
allyears_indexed.loc['F', (claires), :].unstack(level=2)

# fix stacked plot by filling NaNs with zeros, adding labels, setting axis range
# the plt.stack(xrange=N, M*N), M-#ofLines
plt.figure(figsize=(12, 2.5))
plt.stackplot(range(1880, 2019),
              allyears_indexed.loc[('F', claires), :].unstack(level=2).fillna(
                  0),
              labels=claires);

# without fillna(),data will truncated, only shown from the last Nan values
plt.legend(loc='uplter left')
plt.axis(xmin=1880, xmax=2018);

##evaluate len(index)
allyears_indexed.loc[('F', claires), :].unstack(level=1).reset_index().year

allyears = pd.read_csv('allyears.csv.qz')
allyears_byyear = allyears.set_index(['sex', 'year']).sort_index()


def getyear(sex, year):
    return (allyears_byyear.loc[sex, year]  # select M/F, year
            .sort_values('number', ascending=False)  # sort by most common
            .head(10)  # only ten
            .reset_index()  # lose the index
            .name)  # return a name-only Series


# create DataFrame with columns given by top ten name Series for range of years
pd.DataFrame({year: getyear('M', year) for year in range(2010, 2019)})


# @ notation=argument, sex col=sex argument(@)
def plotname(sex, name):
    data = allyears.query('sex == @sex and name == @name')
    plt.plot(data.year, data.number, label=name)
    plt.axis(xmin=1880, xmax=2018)
