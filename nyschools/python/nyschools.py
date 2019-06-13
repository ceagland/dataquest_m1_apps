#!/usr/bin/env python
# coding: utf-8

# # Dataquest - Cleaning Data with Python and Pandas
# 
# Just another one of the courses. Been through this a few times, but it's more practice.
# 
# We're examining/cleaning some data from NY schools.
# 
# There are several data sources, and they can all be found [here](https://opendata.cityofnewyork.us/).
# 
# The data includes:
# - SAT Results
# - AP Results
# - Demographics
# - Graduation Rates
# - Class Sizes
# - Enrollment Size

# #### Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from mpl_toolkits.basemap import Basemap


# #### Import data

# In[2]:


data_files = glob.glob('data/*.csv')


# In[3]:


data_files


# In[4]:


data = {}
for file in data_files:
    datakey = file.replace('.csv','').replace('data\\','')
    data[datakey] = pd.read_csv(file)


# In[5]:


all_survey = pd.read_csv('data/survey_all.txt', delimiter='\t', encoding='windows-1252')
d75_survey = pd.read_csv('data/survey_d75.txt', delimiter='\t', encoding='windows-1252')

survey = pd.concat([all_survey, d75_survey], axis = 0, sort = False)


# #### Examine first few rows of each data set
# 
# The Dataquest instruction is to loop through the keys and print the heads... but I don't like the way it looks. We'll print it all individually.

# In[6]:


data.keys()


# In[7]:


data['sat_results'].head()


# In[8]:


data['ap_2010'].head()


# In[9]:


data['demographics'].head()


# In[10]:


data['class_size'].head()


# In[11]:


data['graduation'].head()


# In[12]:


data['hs_directory'].head()


# In[13]:


survey.head()


# #### Initial thoughts
# 
# All kinds of stuff going on here! That's pretty much it. The focus will be SAT results, but we can use the other datasets to add in different dimensions for comparison.
# 
# Also, I'll definitely need to look at the Data Dictionary for the survey.

# #### Only keep relevant survey columns
# 
# 2,773 columns is a lot. I'll subset down to the relevant ones.

# In[14]:


survey.head()


# In[15]:


retain_cols = ["dbn", "rr_s", "rr_t", "rr_p", "N_s", "N_t", "N_p", "saf_p_11", "com_p_11",
 "eng_p_11", "aca_p_11", "saf_t_11", "com_t_11", "eng_t_11", "aca_t_11", "saf_s_11",
 "com_s_11", "eng_s_11", "aca_s_11", "saf_tot_11", "com_tot_11", "eng_tot_11", "aca_tot_11"]

survey = survey.loc[:, retain_cols]
survey.rename(columns = {'dbn':'DBN'}, inplace = True)

data['survey'] = survey


# #### Creating consistent identifier
# 
# The DBN is a school identifier. In some cases the name is 'dbn' (lower case) or split into multiple columns. Will adjust so each DataFrame has a DBN column.

# In[16]:


data['hs_directory'].rename(columns = {'dbn':'DBN'}, inplace = True)


# In[17]:


data['class_size'].head()


# In[18]:


data['class_size']['DBN'] = data['class_size']['CSD'].astype('str').str.zfill(2) + data['class_size']['SCHOOL CODE']


# In the AP data, there is a single DBN with multiple rows... but, one has missing values. Rather than deal with that manually, I'll remove the rows with missing values and the problem will solve itself. We don't lose anything - when the *AP Test Takers* column is missing, so other the other columns.

# In[19]:


data['ap_2010'] = data['ap_2010'][pd.notnull(data['ap_2010']['AP Test Takers '])]


# #### Creating SAT scores
# 
# Make the columns with the subject SAT scores numeric, then sum them.

# In[20]:


tonumcols = ['SAT Math Avg. Score','SAT Critical Reading Avg. Score','SAT Writing Avg. Score']

for col in tonumcols:
    data['sat_results'].loc[:,col] = pd.to_numeric(data['sat_results'].loc[:,col],errors='coerce')


# In[21]:


data['sat_results'].info()


# In[22]:


data['sat_results']['sat_score'] = data['sat_results'][tonumcols].sum(axis = 1)


# ####  Parse location data
# 
# The hs_directory data includes location data.

# In[23]:


data['hs_directory'].head()


# In[24]:


data['hs_directory']['lat'] = pd.to_numeric(data['hs_directory']['Location 1'].str.extract(r'(\(.+\))').loc[:,0].str.replace('(','').str.replace(')','').str.split(',').str[0], errors = 'coerce')

data['hs_directory']['lon'] = pd.to_numeric(data['hs_directory']['Location 1'].str.extract(r'(\(.+\))').loc[:,0].str.replace('(','').str.replace(')','').str.split(',').str[1], errors = 'coerce')


# In[25]:


data['hs_directory']


# #### Combining the Data
# 
# Now, we start making sure DBN is a unique identifier in each DataFrame before combining them.

# In[26]:


data['class_size'].head()


# Ah, definitely not unique. Looks like there are class size rows by grade and program type. I'll use the high school grades and gen ed classes.

# In[27]:


data['class_size'] = data['class_size'][(data['class_size']['GRADE '] == '09-12')                                         & (data['class_size']['PROGRAM TYPE'] == 'GEN ED')]


# In[28]:


data['class_size'].head()


# In[29]:


data['class_size'] = data['class_size'].groupby('DBN').agg(np.mean)
data['class_size'].reset_index(inplace = True)
data['class_size']['DBN'].value_counts().max() == 1 #Verification of unique


# We averaged some averages... but, down to a single row for each DBN.
# 
# This isn't yet true in the demographics data, which provide annual results. We just want 2011-12 school year, to match the SAT results data.

# In[30]:


data['demographics'] = data['demographics'][data['demographics']['schoolyear'] == 20112012]
data['demographics'].reset_index(drop = True, inplace = True)
data['demographics']['DBN'].value_counts().max() == 1 #Verification of unique


# Finally... same deal with the graduation data set. We'll take the 2006 Total Cohort (last available).

# In[31]:


data['graduation'] = data['graduation'][(data['graduation']['Demographic'] == 'Total Cohort')                                         & (data['graduation']['Cohort'] == '2006')]
data['graduation'].reset_index(drop = True, inplace = True)
data['graduation']['DBN'].value_counts().max() == 1 #Verification of unique


# #### Combine the Data
# 
# It's time! We'll combine all of the data. *sat_results* is the most important DataFrame, so any merging activity will be completed with the intention of retaining as much of *sat_results* as possible while minimizing missing values.
# 
# If this was less exploratory, it'd be a bit nicer to pass the joins as a couple of loops (one for left joins, one for inner joins). But, I want to see the shape after each join to see how many rows I've lost/columns I've gained after each join.

# In[32]:


combined = data['sat_results']
combined.shape


# In[33]:


#Left joins of ap and graduation data (they're missing more DBN values than other DataFrames)
combined = pd.merge(left = combined, right = data['ap_2010'], on = 'DBN', how = 'left')
combined = pd.merge(left = combined, right = data['graduation'], on = 'DBN', how = 'left')
combined.shape


# In[34]:


#Inner joins on the remainder
combined = pd.merge(left = combined, right = data['class_size'], on = 'DBN', how = 'inner')
combined.shape


# In[35]:


combined = pd.merge(left = combined, right = data['demographics'], on = 'DBN', how = 'inner')
combined.shape


# In[36]:


combined = pd.merge(left = combined, right = data['survey'], on = 'DBN', how = 'inner')
combined.shape


# In[37]:


combined = pd.merge(left = combined, right = data['hs_directory'], on = 'DBN', how = 'inner')
combined.shape


# Lost a lot with the hs_directory data set, but still plenty to work with. We've also got a lot of variables now!
# 
# #### Filling Missing Values
# 
# Dataquest suggests filling with means, then 0 for anything still remaining (for columns with no mean available).
# 
# This doesn't handle all the columns that are actually numeric but still stored as strings. However... going to stick with their method to stay consistent. Chances are the columns in question won't be used anyway.

# In[38]:


means = combined.mean()
combined.fillna(means, inplace = True)
combined.fillna(0, inplace = True)
combined.isnull().mean()


# #### Creating a school district variable
# 
# Looks like we don't always have a school district... but it's included in the DBN, so we can use that.

# In[39]:


def extract_two(x):
    return(x[0:2])

combined['school_dist'] = combined['DBN'].apply(extract_two)


# In[40]:


pd.DataFrame(combined['school_dist'].value_counts()).sort_index()


# ## Initial Look at Correlations
# 
# OK, we now have a somewhat complete data set, and it's time to start looking at relationships.
# 
# #### Correlations

# In[41]:


pd.options.display.max_rows = 85 # specific to the correlations view


# In[42]:


correlations = combined.corr()
correlations = correlations['sat_score']
print(correlations.sort_values())


# *frl_percent* has a somewhat strong negative correlation... no idea what it is, though. Let's check the data dictionary and... ah, frl stands for Free and Reduced Lunch. In other words, there is a negative relationship between the proportion of students on the Free and Reduced Lunch program and SAT scores.
# 
# There are several enrollment-related variables showing up with positive correlations (*male_num, female_num, total_enrollment, AVERAGE CLASS SIZE, total_students*). It's hard to say why this is. *AVERAGE CLASS SIZE* is the most surprising to me. It's not a mental leap to think larger schools might end up with more/better resources, but a positive correlation between class size and SAT scores is somewhat surprising.
# 
# The Dataquest lesson says figures for female enrollment (*female_per, female_num*) correlate positively with SAT and males figures correlate negatively... that's not actually what I'm seeing here. The female/male figures are not meaningfully different. There do seem to be some race-based differences.
# 
# The only surprise of the strong correlation between *sat_score* and the subject scores is that they're not stronger than we see. Is something wrong? (*Hint: YES, as I find out in a moment*)
# 
# Positive and negative, there are a lot of correlations in the 0.2-0.4 range.
# 
# I would like to see how *frl_percent* is distributed.

# In[43]:


p = combined.frl_percent.hist()


# In[44]:


combined.frl_percent.value_counts(normalize = True, bins = 6)


# Wow, I didn't realize it was that high. About 43% of schools have at least 72% (round numbers) of their students on free/reduced lunch plans.

# #### Some more relationships (Continuing Correlation)

# In[45]:


#All the different syntaxes for plotting are starting to annoy me.
p = combined.plot.scatter(x = 'total_enrollment', y = 'sat_score')


# Looks pretty weak to me. Those 0 SAT scores aren't so believeable. This might be my fault.

# In[46]:


combined['sat_score'].min()


# Ah, big miss.

# In[47]:


combined[combined['sat_score'] < 100]


# Aha. The component scores are present, but they're all the same. Must have been the mean fill. Didn't recalculate the *sat_score* after. I don't really like the mean fill method. It's nearly 10% of the data. But, it's not *that* much and it's the Dataquest suggestion, so I'll go ahead and keep it.
# 
# First, I'll confirm with the original data that the subject scores actually missing and there's not something else going on. Assuming they are, I'll just recalculate *sat_score*

# In[48]:


data['sat_results'][data['sat_results']['sat_score'] == 0]


# Confirmed.

# In[49]:


tonumcols = ['SAT Math Avg. Score','SAT Critical Reading Avg. Score','SAT Writing Avg. Score'] #Just a reminder
combined['sat_score'] = combined[tonumcols].sum(axis = 1)


# #### If at first you don't succeed, fix your data and redo everything
# 
# Well, the correlations and the one plot that didn't look right.

# In[50]:


correlations = combined.corr()
correlations = correlations['sat_score']
correlations.sort_values()


# Well, this is substantially different (more the strength than the direction/order). This explains why the component score correlations with *sat_score* were so weak before. The race-based differences got enough stronger. *male_per* and *female_per* are now more different from one another, though neither correlations are particularly strong.
# 
# Let's look at the total_enrollment plot again and see if grahpically we now see a stronger correlation.

# In[51]:


p = combined.plot.scatter(x = 'total_enrollment', y = 'sat_score')


# In[52]:


np.mean(combined['sat_score'])


# Well, a little something. It's really variable for the schools with total enrollment under about 1000. From enrollment of 1,000 to about 4,000, SAT scores are at or a bit above the overall average, with a few standouts.
# 
# Let's look at the little batch of schools with really low SAT scores (and low enrollment)

# In[53]:


low_enrollment = combined[(combined['total_enrollment'] < 1000) & (combined['sat_score'] < 1000)]

low_enrollment['School Name']


# Ah... I see the word **International** a lot. Which may mean a lot of English as a second language learners. Let's see how the *ell_percent* (Percentage of english language learner's students per school) variable correlates with SAT scores.

# In[54]:


p = combined.plot.scatter(x = 'ell_percent', y = 'sat_score')


# In[55]:


longitudes = combined['lon'].tolist()
latitudes  = combined['lat'].tolist()


# In[56]:


m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='h'
)

m.drawmapboundary(fill_color='#FFFFFF')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

p = m.scatter(longitudes, latitudes, s = 20, zorder = 2, latlon = True, c = combined['ell_percent'], cmap = 'summer')


# In[57]:


districts = combined.groupby('school_dist').agg(np.mean)
districts.reset_index(inplace = True)
districts.head()


# In[58]:


m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='h'
)

m.drawmapboundary(fill_color='#FFFFFF')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = districts['lon'].tolist()
latitudes  = districts['lat'].tolist()

p = m.scatter(longitudes, latitudes, s = 50, zorder = 2, latlon = True, c = districts['ell_percent'], cmap = 'summer')


# ## Exploratory Analysis
# 
# #### Relationship between SAT scores and perceived safety
# 
# First, we'll look at correlations between *sat_score* and answers to survey responses.

# In[59]:


surveynames = data['survey'].columns.tolist()
surveynames.insert(1,'sat_score')
p = pd.DataFrame(combined.loc[:,surveynames].corr()['sat_score'])     .iloc[1:,:]     .plot(kind = 'bar', legend = False,           title = 'Correlations of Survey Responses with SAT Scores', rot = 45, figsize = (12,5), fontsize = 12)


# Some of the higher correlations are:
# 
# **N_s**  
# Number of Student Respondents. Ah. We did some some evidence that larger schools had higher SAT scores, so this is really just a function of that (we might find otherwise if it happens that these schools have higher survey responses, and we could take it as some other sign of engagement, but chances are it's just school size).
# 
# **N_p**  
# Number of Parent Respondents. Same as *N_s*
# 
# **aca_s_11**  
# Academic expectations score based on student responses. In other words, how did students feel about what they were expected to achieve. So, there is a positive relationship between high expectations and higher SAT scores. I can buy that.
# 
# **saf_s_11 / saf_t_11 / saf_tot_11**  
# Safety and respect based on student/teacher/total responses. In a way, it's nice to see these on the list. It's intuitive that students who feel more safe will be able to concentrate more on academics. The flip side is that students who don't feel safe don't do as well, and we have students who don't feel safe. But, we have the opportunity to identify that using this data, which is part of the benefit.
# 
# These are all positive. There are only a couple of negative correlations, and the strongest is only about -0.1, but since it's the only one in that direction let's talk about it:
# 
# **com_p_11**
# Communicaton score based on parent responses. It's a weak relationship, but still funny to see this here. A negative relationship between how parents feel the school is communicating and SAT scores.
# 
# #### Dig deeper into safety scores
# 
# These show up quite a bit. Let's visualize the relationship beyond just a single figure correlation.

# Investigate safety scores.
# Compute the average safety score for each district.
# Make a map that shows safety scores by district.
# Write up your conclusions about safety by geographic area in a Markdown cell. You may want to read up on the boroughs of New York City.

# In[60]:


s = sns.scatterplot(x = combined['saf_s_11'], y = combined['sat_score'], hue = combined['borough'])


# Well, It's not clear that the safety to sat_score relationship holds up all that well. After safety reaches 7, there is a lot of variation in *sat_scores*. Maybe safety is something that only matters when it doesn't exist. When the score gets past 7, safety becomes less important. There isn't much relationship at all between the 2 variables from 5-7 or from 7-9... some, but not much. It only starts to look stronger as a whole.
# 
# And, it's hard to tell with the colors, but it's not immediately clear that schools from any of the five boroughs are more or less safe on average than others. A map may help with that.

# In[61]:


safetybydistrict = combined.loc[:, ['school_dist','lat','lon','saf_s_11']]
safetybydistrict = safetybydistrict.groupby('school_dist').agg(np.mean)


# In[62]:


m = Basemap(
    projection='merc', 
    llcrnrlat=40.496044, 
    urcrnrlat=40.915256, 
    llcrnrlon=-74.255735, 
    urcrnrlon=-73.700272,
    resolution='h'
)

m.drawmapboundary(fill_color='#FFFFFF')
m.drawcoastlines(color='#6D5F47', linewidth=.4)
m.drawrivers(color='#6D5F47', linewidth=.4)

longitudes = safetybydistrict['lon'].tolist()
latitudes  = safetybydistrict['lat'].tolist()

#combined['district_num'] = combined['school_dist'].astype('category').cat.codes
m = m.scatter(longitudes, latitudes, s = 50, zorder = 2, latlon = True, c = safetybydistrict['saf_s_11'], cmap = 'Blues')


# Where darker is more safe, it doesn't look like there's any completely obvious split between boroughs. Brooklyn/Queens to the bottom right of the map do have a few of the less-safe school districts.
# 
# This is partly a result of the "mean by school district" look. Below, I show what happens... there's not a lot of variation from one school district to another. The range of the means is only about 1.25 points (on the 10 point survey scale).

# In[63]:


safetybydistrict.saf_s_11.describe()


# #### Relationship between SAT scores and race

# In[64]:


pd.DataFrame(combined[['sat_score','white_per','asian_per','black_per','hispanic_per']].corr()['sat_score'])     .iloc[1:,:]     .plot(kind = 'bar', legend = False, title = 'Correlations between Race and SAT scores', rot = 0, fontsize = 12)     .tick_params(bottom = False)


# We'd done some passive review of these earlier. There is a somewhat strong positive correlation with SAT scores and a school's proportion of white/asian students, and a negative correlation between SAT scores and proportion of black/hispanic students.
# 
# We'll explore *hispanic_per* below.

# In[65]:


s = sns.regplot(x = combined['hispanic_per'], y = combined['sat_score'])


# Here I've added a regression line. As we'd expect given the correlation we saw, the line has a negative slope. There are some schools below about 25% hispanic students with high SAT scores.  After 25% things steady off, with what looks like a less steep decrease (just looking at the points - ignoring the line here) up to just short of 100%. The schools at 100% have uniformly low SAT scores. Let's find out more about those schools.

# In[66]:


combined.loc[combined['hispanic_per'] > 95, ['school_name','sat_score', 'hispanic_per']]


# I looked for more information about a couple of these schools.
# 
# **Multicultural High School**  
# 
# The name is something of a misnomer, as virtually every student at the school is Hispanic. Many of the students at the school are foreign-born and English is not their first language.
# 
# **International School for Liberal Arts**  
# 
# Despite low test scores and many International students, I found two bits of information I think were interesting, if not informative.
# 
# 1 - Despite test scores being low across subjects (and outside of just the SATs), only 14% of the students are actually taking SATs. I've explored data in the past where schools with low SAT participation tended to have higher average SAT scores, because students who would do well were encouraged to take it and other students were discouraged. This is why state level results must be taken with a grain of salt - some states require that all students take the SATs. Those states tend to have lower average SAT scores. So, it was surprising to me that the scores were low with this level of participation, but of course the international/language factor is a likely influence.
# 
# 2 - This doesn't help with the data analysis much, but despite scoring low in academics, this school scores highly (on greatschools, anyway) for Academic Progress. So those low academic scores are for students that were not doing as well the year prior.
# 
# Now I'll do the same thing for schools with high SAT scores and a low proportion of hispanic students. How do these schools differ?

# In[67]:


combined.loc[(combined['hispanic_per'] < 10) & (combined['sat_score'] > 1800), ['school_name','sat_score','hispanic_per']]


# These are specialized high schools in New York. Every single one is selective, requiring students to score highly on an admissions test. A couple of them even have *Reputation* sections on Wikipedia.

# #### Relationship between SAT scores and gender

# In[68]:


pd.DataFrame(combined[['sat_score','male_per','female_per']].corr()['sat_score']).iloc[1:,:]     .plot(kind = 'bar', legend = False, title = 'Correlations between Gender and SAT scores', rot = 0, fontsize = 12)     .tick_params(bottom = False)


# Though these relationships are fairly week, we see a positive relationship between a school having a higher proportion of female students and SAT scores.
# 
# Plotting both of these works for a visual, but each of male_per and female_per is 1 minus the other, so the relationship has equal strength in either direction.

# In[69]:


s = sns.regplot(x = combined['female_per'], y = combined['sat_score'])


# Not the most compelling evidence. Most schools fall in the 40-60 pct gender proportion range. There are clusters of schools with high SAT scores that have majority male or female students. The all-female schools do look slightly above average as a group, but only slightly.

# In[70]:


combined.loc[(combined['female_per'] > 60) & (combined['sat_score'] > 1700), ['school_name','sat_score','female_per']]


# Again, the high performing schools are the selective schools. Townsend Harris High School is ranked (if you like the US News & World Report) as the top school in the state, and its admission rate is about half that of Harvard's.

# #### Relationship between SAT scores and Advanced Placement
# 
# Calculate the percentage of students in each school that took an AP exam.
# Divide the AP Test Takers column by the total_enrollment column.
# The column name AP Test Takers has a space at the end -- don't forget to add it!
# Assign the result to the ap_per column.
# Investigate the relationship between AP scores and SAT scores.
# Make a scatter plot of ap_per vs. sat_score.
# What does the scatter plot show? Record any interesting observations in a Markdown cell.

# In[71]:


combined['ap_per'] = combined['AP Test Takers '] / combined['total_enrollment']


# In[75]:


s = sns.regplot(x = combined['ap_per'], y = combined['sat_score'])


# Not much of a *linear* relationship. Well, not much of a recognizable relationship of any kind. When over 50% or so of students take AP exams, the schools seem to score about 1200 on the SATs. Seems like some schools might be a bit more focused on having students take AP exams.
# 
# I am curious about the cluster with great than 40% of students taking AP exams and getting high SAT scores. Almost certainly the competitive specialized schools again. I'll check this first because I expect the result will be familiar.

# In[76]:


combined.loc[(combined['sat_score'] > 1800) & (combined['ap_per'] > 0.4), ['school_name','sat_score','ap_per']]


# Yes, this is almost the same the the list of schools with high SAT scores and a low proportion of hispanic students explored above in the section on race correlations with the SAT score.
# 
# Next, looking at schools with high *ap_per*

# In[79]:


combined.loc[combined['ap_per'] > 0.70, ['school_name','sat_score','ap_per']]


# Interesting. Some of these are selective, but not in the way we've seen. The Frances Perkins Academy is mostly "economically disadvantaged" students (wording from US News). The Brooklyn Academy of Global Finance does have an application process, but it's not on the same level as the schools producing much higher SAT scores.
