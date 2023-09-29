#!/usr/bin/env python
# coding: utf-8

# # Descriptip Statistics

# # Central tendency , When we talk about average value, middle value and most frequent value it means we informally talking about mean, median and mode

# In[511]:


import numpy as np
from scipy import stats


# In[512]:


df = np.array([39,29,43,52,39,44,40,31,44,35])


# In[513]:


df


# # Mean

# In[514]:


# Calculate mean
df_mean = np.mean(df)


# In[515]:


print("The mean is : \n", df_mean)


# # Median

# In[516]:


# Calculate median
df_median = np.median(df)


# In[517]:


print("The median is : \n", df_median)


# # Mode

# # It is the value that has the highest frequency in the given data set. The data set may have no mode if the frequency of all data points is the same. Also, we can have more than one mode if we encounter two or more data points having the same frequency.

# In[518]:


# Calculate mode
df_mode = stats.mode(df)


# In[519]:


print("The mode is : \n", df_mode)


# # Real dataset

# In[520]:


import pandas as pd
data = pd.read_csv(r"C:\Users\mk744\Downloads\documents\datafiles\Example.csv")


# In[521]:


data


# # Find the mean of Satisfaction attributes

# In[522]:


data_mean = np.mean(data['Satisfaction'])


# In[523]:


print("The mean of Satisfaction is given : \n", data_mean)


# # Find the mean of Loyality attributes

# In[524]:


data_mean_1 = np.mean(data['Loyalty'])


# In[525]:


print("The mean of Loyalty is given :\n %.3f"  %data_mean_1)


# # Find the median of Satisfaction attributes

# In[526]:


data_median = np.median(data['Satisfaction'])


# In[527]:


print("The median of Satisfaction is given by : \n", data_median)


# # Find the median of Loyalty attributes

# In[528]:


data_median_1 = np.median(data['Loyalty'])


# In[529]:


print("The median of Loyalty is given by : \n %.3f" %data_median_1)


# # Find the mode of Satisfaction attributes

# In[530]:


data_mode = stats.mode(data['Satisfaction'])


# In[531]:


print("The mode of Satisfaction is given by : \n", data_mode)


# # Find the mode of Loyalty attributes

# In[532]:


data_mode_1 = stats.mode(data['Loyalty'])


# In[533]:


print("The mode of Loyalty is given by : \n", data_mode_1)


# # Variation and Shape

# # Range
# <!-- The range describes the difference between the largest and smallest data point in our data set. The bigger the range, the more the spread of data and vice versa. -->

# # The range describes the difference between the largest and smallest data point in our data set. The bigger the range, the more the spread of data and vice versa.

# In[534]:


Maximum = max(df)
Minimum = min(df)


# # Calculate Range

# In[535]:


Range = Maximum - Minimum 


# In[536]:


print("Maximum = {} , Minimum = {}, Range = {}".format(Maximum, Minimum, Range))


# # Using Data set

# In[537]:


Maxi = max(data['Satisfaction'])
Mini = min(data['Satisfaction'])


# # Calculate Range

# In[538]:


Ranges = Maxi - Mini


# In[539]:


print("Maxi = {}, Mini = {}, Ranges = {}".format(Maxi, Mini, Ranges))


# # Find Range

# In[540]:


dataset = np.array([29, 31, 35, 39, 40, 43, 44, 44, 52])


# In[541]:


dataset


# In[542]:


MAXI = max(dataset)
MINI = min(dataset)


# In[543]:


RANGE = MAXI - MINI


# In[544]:


print("MAXI = {}, MINI = {}, RANGE = {}".format(MAXI, MINI, RANGE))


# # Variance

# In[545]:


import statistics


# In[546]:


print("Var = ", (statistics.variance(dataset) ))


# In[547]:


print("Variance = ", (statistics.variance(df)))


# In[548]:


print("Variance = ", (statistics.variance(data['Satisfaction'])))


# In[549]:


print("Variance = ", (statistics.variance(data['Loyalty'])))


# # Std

# In[550]:


Var =  (statistics.stdev(data['Satisfaction']))
print("Std = ", Var)


# In[551]:


Var_1 = (statistics.stdev(data['Loyalty']))
print("Std = ", Var_1)


# # Coefficient of Variation of Satisfaction

# In[552]:


coefficient_of_Var = (Var/data_mean)*100


# In[553]:


print(f"The coefficient of variation : {coefficient_of_Var : .2f}%")


# # Coefficient of Variation of Loyality

# In[554]:


Coefficient_of_Var = (Var_1 / data_mean_1) * 100


# In[555]:


print(f"The coefficient of variation : {Coefficient_of_Var : .2f}%")


# # Z-Score:=> It is used to find the outliers in the given dataset, if the Z-Score is -3.0 and +3.0 then it is indicating there is an outliers present in the dataset

# In[556]:


# Find the z score of the given data
df


# In[557]:


Std_dev = (statistics.stdev(df))
print("Std dev = ", Std_dev)


# In[558]:


X1 = 39


# In[559]:


Z_Score = (X1 - df_mean) / Std_dev
print(f"The Z-Score : {Z_Score : .2f}%")


# In[560]:


X2 = 29
Z_Score_1 = (X2 - df_mean) / Std_dev
print(f"The Z-Score : {Z_Score_1 : .2f}%")


# In[561]:


X3 = 43
Z_Score_3 = (X3 - df_mean)/Std_dev
print(f"The Z-Score : {Z_Score_3 : .2f}%")


# # Using Real Dataset

# In[562]:


OUTLIERS = pd.read_csv(r"C:\Users\mk744\Downloads\documents\datafiles\birth.csv")
OUTLIERS.head()


# In[563]:


OUTLIERS.dtypes


# #  Outliers in the "year" Columns

# In[564]:


out=[]
def Zscore_outlier(dbase):
    m = np.mean(dbase)
    sd = np.std(dbase)
    for i in dbase: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(OUTLIERS['year'])


# #  Outliers in the "month" columns

# In[565]:


out = []
def Z_Score_Outliers(demo):
    mean_value = np.mean(demo)
    std_dev = np.std(demo)
    for i in demo:
        Z_Score = (i - mean_value) / std_dev
        if np.abs(Z_Score) > 3:
            out.append(i)
    print("Outliers:", out)
Z_Score_Outliers(OUTLIERS['month'])


# # Outliers in the "day" columns

# In[566]:


out = []
def Z_Score_Outliers(demo):
    mean_value = np.mean(demo)
    std_dev = np.std(demo)
    for i in demo:
        Z_Score = (i - mean_value) / std_dev
        if np.abs(Z_Score) > 3:
            out.append(i)
    print("Outliers:", out)
Z_Score_Outliers(OUTLIERS['day'])


# # Outliers in the "births" columns

# In[567]:


out = []
def Z_Score_Outliers(demo):
    mean_value = np.mean(demo)
    std_dev = np.std(demo)
    for i in demo:
        Z_Score = (i - mean_value) / std_dev
        if np.abs(Z_Score) > 3:
            out.append(i)
    print("Outliers:", out)
Z_Score_Outliers(OUTLIERS['births'])


# # Visualize the outliers, Data visualization is useful for data cleaning, exploring data, detecting outliers identifying trends and clusters

# In[568]:


# Importing libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import warnings
warnings.filterwarnings('ignore')


# # Box Plot

# In[569]:


def Box_plots(df):
    plt.figure(figsize=(10, 4))
    plt.title("Box Plot")
    sns.boxplot(df)
    plt.show()
Box_plots(OUTLIERS['year'])
plt.show()


# In[570]:


def Box_plots(df):
    plt.figure(figsize=(10, 4))
    plt.title("Box Plot")
    sns.boxplot(df)
    plt.show()
Box_plots(OUTLIERS['day'])
plt.show()


# In[571]:


def Box_plots(df):
    plt.figure(figsize=(10, 4))
    plt.title("Box Plot")
    sns.boxplot(df)
    plt.show()
Box_plots(OUTLIERS['births'])
plt.show()


# # Histogram Plot

# In[572]:


def hist_plots(df):
    plt.figure(figsize=(10, 4))
    plt.hist(df)
    plt.title("Histogram Plot")
    plt.show()
hist_plots(OUTLIERS['year'])


# In[573]:


def hist_plots(df):
    plt.figure(figsize=(10, 4))
    plt.hist(df)
    plt.title("Histogram Plot")
    plt.show()
hist_plots(OUTLIERS['day'])


# In[574]:


def hist_plots(df):
    plt.figure(figsize=(10, 4))
    plt.hist(df)
    plt.title("Histogram Plot")
    plt.show()
hist_plots(OUTLIERS['births'])


# # Scatter Plot

# In[575]:


# Relationship between year and day
def scatter_plots(df1,df2):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(df1,df2)
    ax.set_xlabel('year')
    ax.set_ylabel('day')
    plt.title("Scatter Plot")
    plt.show()
scatter_plots(OUTLIERS['year'],OUTLIERS['day'])


# In[576]:


# Relationship between year and births
def scatter_plots(df1,df2):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(df1,df2)
    ax.set_xlabel('year')
    ax.set_ylabel('day')
    plt.title("Scatter Plot")
    plt.show()
scatter_plots(OUTLIERS['year'],OUTLIERS['births'])


# In[577]:


# Relationship between day and births
def scatter_plots(df1,df2):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(df1,df2)
    ax.set_xlabel('year')
    ax.set_ylabel('day')
    plt.title("Scatter Plot")
    plt.show()
scatter_plots(OUTLIERS['day'],OUTLIERS['births'])


# # Distribution Plot

# In[578]:


def dist_plots(df):
    plt.figure(figsize=(10, 4))
    sns.distplot(df)
    plt.title("Distribution plot")
    sns.despine()
    plt.show()
dist_plots(OUTLIERS['day'])


# In[579]:


def dist_plots(df):
    plt.figure(figsize=(10, 4))
    sns.distplot(df)
    plt.title("Distribution plot")
    sns.despine()
    plt.show()
dist_plots(OUTLIERS['year'])


# In[580]:


def dist_plots(df):
    plt.figure(figsize=(10, 4))
    sns.distplot(df)
    plt.title("Distribution plot")
    sns.despine()
    plt.show()
dist_plots(OUTLIERS['births'])


# # Normal QQPlots

# In[581]:


def qq_plots(df):
    plt.figure(figsize=(10, 4))
    qqplot(df,line='s')
    plt.title("Normal QQPlot")
    plt.show()
qq_plots(OUTLIERS['year'])


# In[582]:


def qq_plots(df):
    plt.figure(figsize=(10, 4))
    qqplot(df,line='s')
    plt.title("Normal QQPlot")
    plt.show()
qq_plots(OUTLIERS['day'])


# In[583]:


def qq_plots(df):
    plt.figure(figsize=(10, 4))
    qqplot(df,line='s')
    plt.title("Normal QQPlot")
    plt.show()
qq_plots(OUTLIERS['births'])


# # Kurtosis = 3 corresponds to a normal distribution with no excess kurtosis.  Kurtosis > 3 indicates heavy tails (leptokurtic) with more extreme values. Kurtosis < 3 indicates light tails (platykurtic) with fewer extreme values.

# #  Kurtosis is a statistical measure that quantifies the "tailedness" or the shape of the probability distribution of a dataset. In simpler terms, it tells you whether the data is more or less extreme (outliers) than a normal distribution. Kurtosis provides information about the presence and degree of outliers in your data and how these outliers deviate from a normal distribution.

# In[584]:


OUTLIERS['year'].kurt()


# In[585]:


OUTLIERS['day'].kurt()


# In[586]:


OUTLIERS['births'].kurt()


# # Skewness measures the extent to which the data values are not symmetrical around the mean 

# # 1>   Mean < Median : negative, or left-Skewned distribution.                              2> Mean = Median : Symmetrical Distribution (zero Skewness).                          3> Mean > Median : positive , or right - Skewed distribution.

# In[587]:


OUTLIERS['year'].skew()


# In[588]:


OUTLIERS['day'].skew()


# In[589]:


OUTLIERS['births'].skew()


# # Visualisation of skewness 

# In[590]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Find skewness
data = OUTLIERS['year']

# Calculate skewness using SciPy's skew function
skewness = stats.skew(data)

# Create a histogram
plt.hist(data, bins=30, alpha=0.7, color='#108A99', label='Data')

# Add a vertical line for the mean, median, and mode
plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(np.median(data), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(stats.mode(data).mode[0], color='purple', linestyle='dashed', linewidth=2, label='Mode')

plt.title(f"Skewness = {skewness:.2f}")
plt.legend()
plt.show()


# In[591]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Find skewness
data = OUTLIERS['day']

# Calculate skewness using SciPy's skew function
skewness = stats.skew(data)

# Create a histogram
plt.hist(data, bins=30, alpha=0.7, color='#108A99', label='Data')

# Add a vertical line for the mean, median, and mode
plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(np.median(data), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(stats.mode(data).mode[0], color='purple', linestyle='dashed', linewidth=2, label='Mode')

plt.title(f"Skewness = {skewness:.2f}")
plt.legend()
plt.show()


# In[592]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Find skewness
data = OUTLIERS['births']

# Calculate skewness using SciPy's skew function
skewness = stats.skew(data)

# Create a histogram
plt.hist(data, bins=30, alpha=0.7, color='#108A99', label='Data')

# Add a vertical line for the mean, median, and mode
plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(np.median(data), color='green', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(stats.mode(data).mode[0], color='purple', linestyle='dashed', linewidth=2, label='Mode')

plt.title(f"Skewness = {skewness:.2f}")
plt.legend()
plt.show()


# # Visualisation of skewness on different plots

# In[600]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Generate example data with different skewness values
data1 = OUTLIERS['year']  # Normally distributed (skewness ≈ 0)
data2 = OUTLIERS['day']  # Positively skewed (right-skewed)
data3 = OUTLIERS['births']  # Negatively skewed (left-skewed)

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot distribution plots for each dataset
sns.histplot(data1, kde=True, ax=axes[0], color='blue', label='Data 1')
axes[0].set_title(f"Skewness = {skew(data1):.2f}")

sns.histplot(data2, kde=True, ax=axes[1], color='green', label='Data 2')
axes[1].set_title(f"Skewness = {skew(data2):.2f}")

sns.histplot(data3, kde=True, ax=axes[2], color='purple', label='Data 3')
axes[2].set_title(f"Skewness = {skew(data3):.2f}")

# Add a common legend
for ax in axes:
    ax.legend()

plt.tight_layout()
plt.show()


# # Visualisation of Kurtosis

# In[596]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
data_1 = OUTLIERS['year']
# Calculate kurtosis for each dataset
kurtosis1 = kurtosis(data_1)
# Create histograms for each dataset
plt.figure(figsize=(15, 6))
plt.subplot(131)
plt.hist(data_1, bins=30, alpha=0.7, color='#108A99', label='Data 1')
plt.axvline(np.mean(data_1), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.title(f"Kurtosis = {kurtosis1:.2f}")
plt.legend()
plt.show()


# In[597]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
data_1 = OUTLIERS['day']
# Calculate kurtosis for each dataset
kurtosis1 = kurtosis(data_1)
# Create histograms for each dataset
plt.figure(figsize=(15, 6))
plt.subplot(131)
plt.hist(data_1, bins=30, alpha=0.7, color='#108A99', label='Data 1')
plt.axvline(np.mean(data_1), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.title(f"Kurtosis = {kurtosis1:.2f}")
plt.legend()
plt.show()


# In[598]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
data_1 = OUTLIERS['births']
# Calculate kurtosis for each dataset
kurtosis1 = kurtosis(data_1)
# Create histograms for each dataset
plt.figure(figsize=(15, 6))
plt.subplot(131)
plt.hist(data_1, bins=30, alpha=0.7, color='#108A99', label='Data 1')
plt.axvline(np.mean(data_1), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.title(f"Kurtosis = {kurtosis1:.2f}")
plt.legend()
plt.show()


# # Visualisation of kurtosis on different plots

# In[602]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
dataset_1 = OUTLIERS['year']  # Normal distribution (kurtosis = 3)
dataset_2 = OUTLIERS['day']  # Laplace distribution (excess kurtosis > 3)
dataset_3 = OUTLIERS['births']  # Uniform distribution (excess kurtosis < 3)

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 8))

# Plot distribution plots for each dataset
sns.histplot(dataset_1, kde=True, ax=axes[0], color='#108A99', label='Data 1')
axes[0].set_title(f"Kurtosis = {kurtosis(dataset_1):.2f}")

sns.histplot(dataset_2, kde=True, ax=axes[1], color='#069AF3', label='Data 2')
axes[1].set_title(f"Kurtosis = {kurtosis(dataset_2):.2f}")

sns.histplot(dataset_3, kde=True, ax=axes[2], color='#FF4500', label='Data 3')
axes[2].set_title(f"Kurtosis = {kurtosis(dataset_3):.2f}")

# Add a common legend
for ax in axes:
    ax.legend()

plt.tight_layout()
plt.show()


# In[608]:


OUTLIERS.head()


# In[605]:


demo = OUTLIERS['year']
demo.hist(ax = None, histtype = 'stepfilled', bins = 20)


# In[606]:


demo_1 = OUTLIERS['month']
demo_1.hist(by = None , histtype = 'stepfilled' , bins = 20)


# In[607]:


demo_2 = OUTLIERS['day']
demo_2.hist(by = None, histtype = 'stepfilled', bins = 20)


# In[615]:


demo_01 = OUTLIERS['gender']
demo_01.hist(by = None, histtype = 'stepfilled', bins = 20)


# In[609]:


from sklearn.preprocessing import OneHotEncoder


# In[629]:


# if you will use sparse = False the remove the toarray()
ohe = OneHotEncoder(drop = 'first')


# In[630]:


demo_02 = ohe.fit_transform(OUTLIERS[['gender']]).toarray()
demo_02


# In[631]:


def Box_plots(df):
    plt.figure(figsize=(10, 4))
    plt.title("Box Plot")
    sns.boxplot(df)
    plt.show()
Box_plots(demo_02)
plt.show()


# In[633]:


out = []
def Z_Score_Outliers(demo):
    mean_value = np.mean(demo)
    std_dev = np.std(demo)
    for i in demo:
        Z_Score = (i - mean_value) / std_dev
        if np.abs(Z_Score) > 3:
            out.append(i)
    print("Outliers:", out)
Z_Score_Outliers(demo_02)


# In[636]:


def dist_plots(df):
    plt.figure(figsize=(10, 4))
    sns.distplot(df)
    plt.title("Distribution plot")
    sns.despine()
    plt.show()
dist_plots(demo_02)


# In[ ]:




