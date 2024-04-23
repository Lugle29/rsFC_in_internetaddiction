# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import stats
import functions as fn
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

################################# Demographics ################################
# Get Data
data_prep = fn.get_prepared_data()
data_unprep = fn.get_behav_connec()
data_unprep = data_unprep.iloc[:,0:4]
data_unprep.describe()

# Age and sex
## Calculate the percentage distribution of age bins
age_percentage = data_unprep['age'].value_counts(normalize=True).sort_index() * 100
## Calculate the gender distribution for the entire dataset
gender_percentage = data_unprep['sex'].value_counts(normalize=True) * 100

print('Age Percentage:')
print(age_percentage)
print('\nGender Percentage:')
print(gender_percentage)

## Examine IAT distribution
IAT_norm = stats.shapiro(data_prep['IAT'])
## Shapiro-Wilk test for normality
print('Shapiro-Wilk Test')
print('Test Statistic:', IAT_norm.statistic)
print('p-value:', IAT_norm.pvalue)

# Check for differences in distribution between Age and Sex
## Conduct Kolmogorov-Smirnov-Test for MW-U Prerequisits
### Perform the Kolmogorov-Smirnov test
statistic, p_value = stats.kstest(data_prep[data_prep['sex'] == 0]['age'],
                                  data_prep[data_prep['sex'] == 1]['age'])
### Print the results
print('Kolgomorov-Smirnov Test')
print('Test Statistic:', statistic)
print('p-value:', p_value)

## Conduct MW-U Test to test for differences on age distribution over gender
## In prepared data women are coded as 1 and men are coded as 0

u_statistic, p_value = stats.mannwhitneyu(data_prep[data_prep['sex'] == 0]['age'],
                                          data_prep[data_prep['sex'] == 1]['age'])
### Print results
print('Mann-Whitney U Test')
print('Test statistic:', u_statistic)
print('p-value:', p_value)

# Conduct linear regression for descriptive purposes of the effect of age and gender on IAT
### Fit Model
# Create design matrix X and target variable y
X = data_prep[['age', 'sex']]
y = data_prep['IAT']
# Add constant term for the intercept
X = sm.add_constant(X)
# Fit the linear regression model
model = sm.OLS(y, X).fit()
# Print the summary of the regression model
print(model.summary())
model.params

# Residuals
residuals = model.resid

## Prerequisits
### Shapiro-Wilk test for normality of residuals
shapiro_test = stats.shapiro(residuals)
print('Shapiro-Wilk test')
print('Test statistic:', shapiro_test[0])
print('p-value:', shapiro_test[1], '\n')

### Perform the White test for heteroscasticity
white_test = sm.stats.diagnostic.het_white(residuals, X)
print('White test')
print('Test statistic:', white_test[0])
print('p-value:', white_test[1])
print('scores:', white_test[2])
print('expected values:', white_test[3])

### Diagnostic Plots
### Plot for linearity
def lin_plot_age(save = False):
    # set style
    sns.set_style('whitegrid')
    
    # plot
    sns.regplot(x = X['age'], y = y,
                line_kws = {'color': 'darkorange', 'alpha': 0.3})
    sns.despine(left=True)
    plt.xlabel(xlabel='Age')
    
    # save
    if save:
        plt.savefig('plots//lin_plot_age.png', dpi = 300)
        
lin_plot_age()

### Plot for normality of the residuals
def qq_plot(residuals = residuals, save = False):
    # Create the Q-Q plot
    qqplot(residuals, line='s')  # 's' is for standardized line
    
    # save
    if save:
        plt.savefig('plots//qq_plot.png', dpi =  300)
        
qq_plot()

# Plot Age by Group
def plot_age(data=data_unprep, save = False):
    data = data.sort_values(by='age')
    # Change the size (height) of the plot
    plt.figure(figsize= (15,8))  # Adjust the figure size as needed
    
    sns.histplot(data=data, x ='age', hue='sex', multiple='dodge', palette=sns.color_palette('Set2'), shrink=.8)
    
    # Customize the plot
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    
    # save
    if save:
        plt.savefig('plots//age_plot.png', dpi = 300)
    plt.show()
    
plot_age()

def plot_IAT(data=data_unprep, save = False):
    # set style
    plt.figure(figsize= (15,8))  # Adjust the figure size as needed
    
    # plot
    sns.histplot(data=data, x ='IAT', kde=True, discrete=True, alpha=0.7)
    
    # customize the plot
    plt.xlabel('IAT')
    plt.ylabel('Count')
    
    if save:
        plt.savefig('plots//IAT_plot.png', dpi = 300)
    plt.show()    
        
plot_IAT()

###################### Descriptive Analysis of FC values ######################
### Default
fc = fn.wrap_up_connectivity('dict')
fc = fc['DF']
avg_corr = sum(fc)/len(fc)

### Plot Correlation Table
def plot_fc(corr_mat = avg_corr, save = False):
    
    plt.figure(figsize=(15,12), dpi = 480)
        
    # Plot the heatmap with annotation
    heatmap = sns.heatmap(corr_mat, annot=True, fmt='.2f')
    
    # Get the color bar object
    cbar = heatmap.collections[0].colorbar
    
    # Set label and adjust its font size
    cbar.set_label(r'$M_r$', fontsize=25)
    
    # Save the plot
    if save:
        plt.savefig('plots//corr.png')
    plt.show()
    
plot_fc(save = True)

### FC means for each predictor group
fc_df = fn.wrap_up_connectivity('df')
fc_df = fc_df.drop(columns = ['ID'])

# Different calculation method. 
def get_avg_fc (data = fc_df, absolute = False):
    '''
    Function to compute average FC strength for between and within networks. 
    Besides default average also absolute average can be computed

    Parameters
    ----------
    data : DataFrame, optional
        Containing all connectivities (columns) for each participant (rows).
        The default is fc_df.
    absolute : bool, optional
        Possibility to get average of absolute connectivity.
        The default is False.

    Returns
    -------
    avg_fc : DataFrame
        Containing the mean and SD for the total, inner and outer connectivity
        of networks.

    '''
    # Prepare dictionary
    avg_fc = {'network':[], 'mean':[], 'sd':[]}
    
    # Get predictor names for within-connections
    predictor_groups = {}
    txt_files = ['reward.txt', 'self_reference.txt', 'attention.txt']
    
    for txt_file in txt_files:
        file_path = os.path.join('data',txt_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            group_name = os.path.splitext(os.path.basename(txt_file))[0]
            predictor_groups[group_name] = [column.strip() for column in lines]
    
    # Get predictor names for without-connections        
    ## Convert lists to sets to remove duplicates
    all_columns = set(data.columns)
    
    set1 = set(predictor_groups['reward'])
    set2 = set(predictor_groups['self_reference'])
    set3 = set(predictor_groups['attention'])
    
    ## Merge sets
    inner_cols = set1.union(set2, set3)
    outer_cols = all_columns - inner_cols
    columns    = {'total':data.columns.tolist(),
                  'inner':list(inner_cols),
                  'outer':list(outer_cols)}
    
    # Get Average FC across participants
    data = data.mean()
    
    # Transform data if absolute values should be calculated
    if absolute == True:
        data = abs(data)
    
    # Average over total, inner and outer
    for conn in columns.keys():
        avg_fc['network'].append(conn)
        avg_fc['mean'].append(np.mean(data[columns[conn]]))
        avg_fc['sd'].append(np.nanstd(data[columns[conn]])) 
    
    for group in predictor_groups.keys():
        avg_fc['network'].append(group)
        avg_fc['mean'].append(np.mean(data[list(predictor_groups[group])]))
        avg_fc['sd'].append(np.nanstd(data[list(predictor_groups[group])]))
        
    return pd.DataFrame(avg_fc)

avg_fc = get_avg_fc()
abs_avg_fc = get_avg_fc(absolute=True)

print('The average FC is:')
print(avg_fc)
print('\n The absolute average FC is:')
print(abs_avg_fc)
