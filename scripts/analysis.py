# -*- coding: utf-8 -*-
"""
@ filename: analysis
@ author: l.glenzer
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import functions as fn
from scipy import stats
import statistics as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

################################# Demographics ################################
# Get Data
data_prep   = fn.get_prepared_data()
data_unprep = fn.get_behav_connec()
data_unprep = data_unprep.iloc[:,0:4]

# Data description
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
def lin_plot_age(X = X, y = y, save = False):
    '''
    Function for linearity plot of the relation Age-IAT

    Parameters
    ----------
    X : DataFrame, optional
        Predictors for linear model. The default is X.
    y : Pandas Series, optional
        Outcome of linear model. The default is y.
    save : bool, optional
        decide, if plot should be saved as .PNG. The default is False.

    Returns
    -------
    None.

    '''
    # Preparation -------------------------------------------------------------
    # set style
    sns.set_style('whitegrid')
    
    # Make Plot ---------------------------------------------------------------
    # Regressionplot
    sns.regplot(x = X['age'], y = y,
                line_kws = {'color': 'darkorange', 'alpha': 0.3})
    
    # Customise Plot
    sns.despine(left=True)
    plt.xlabel(xlabel='Age')
    
    # Save
    if save:
        plt.savefig('plots//lin_plot_age.png', dpi = 300)
        
    # Return None -------------------------------------------------------------
    return
        
lin_plot_age()

### Plot for normality of the residuals
def qq_plot(residuals = residuals, save = False):
    '''
    QQ-Plot to check for normality of the residuals

    Parameters
    ----------
    residuals : Pandas Series, optional
        Containing the residuals of the actual model.
        The default is residuals.
    save : bool, optional
        decide, if plot should be saved as .PNG. The default is False.

    Returns
    -------
    None.

    '''
    # Make Plot ---------------------------------------------------------------
    # Create the Q-Q plot
    qqplot(residuals, line='s')  # 's' is for standardized line
    
    # Save
    if save:
        plt.savefig('plots//qq_plot.png', dpi =  300)
       
    # Return None -------------------------------------------------------------
    return

qq_plot()

# Plot Age by Group
def plot_age(data=data_unprep, save = False):
    '''
    Function to plot the distribution of age stratified by sex.

    Parameters
    ----------
    data : DataFrame, optional
        Containing the relevant columns age and sex. 
        The default is data_unprep.
    save : bool, optional
        decide, if plot should be saved as .PNG. The default is False.

    Returns
    -------
    None.

    '''
    # Preparation -------------------------------------------------------------
    data = data.sort_values(by='age')
    # Change the size (height) of the plot
    plt.figure(figsize= (15,12))  
    
    # Make Plot ---------------------------------------------------------------
    # Histogram
    sns.histplot(data=data,
                 x ='age',
                 hue='sex',
                 multiple='dodge',
                 palette=sns.color_palette('Set2'),
                 shrink=.8,
                 legend=False)
    
    # Customize the plot
    plt.xlabel('Age (years)', fontsize = 18)
    plt.ylabel('Number of Participants', fontsize = 18)
    
    # Increase the size of numbers on the ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Save --------------------------------------------------------------------
    if save:
        plt.savefig('plots//age_plot.png', dpi = 300)
    plt.show()
    
    # Return None -------------------------------------------------------------
    return
    
plot_age(save = True)

def plot_IAT(data=data_unprep, save = False):
    '''
    Function to plot the distribution of IAT-values including a KDE

    Parameters
    ----------
    data : DataFrame, optional
        Including column for IAT values. The default is data_unprep.
    save : bool, optional
        decide, if plot should be saved as .PNG. The default is False.

    Returns
    -------
    None.

    '''
    # Preparation -------------------------------------------------------------
    # Adjust the figure size
    plt.figure(figsize= (15,12))  
    
    # Make Plot ---------------------------------------------------------------
    # Histogram
    sns.histplot(data=data, x ='IAT', kde=True, discrete=True, alpha=0.7)
    
    # Customize the plot
    plt.xlabel('IAT', fontsize = 18)
    plt.ylabel('Number of Participants', fontsize = 18)
    
    # Increase the size of numbers on the ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Save Figure
    if save:
        plt.savefig('plots//IAT_plot.png', dpi = 300)
    # Show Figure
    plt.show()  
    
    # Return None -------------------------------------------------------------
    return
        
plot_IAT()

###################### Descriptive Analysis of FC values ######################
### Default
fc = fn.wrap_up_connectivity('dict')
fc = fc['DF']
avg_corr = sum(fc)/len(fc)

### Plot Correlation Table
def plot_fc(corr_mat = avg_corr, save = False):
    '''
    Plot average connectivity strength in heatmap.

    Parameters
    ----------
    corr_mat : DataFrame, optional
        Correlation-Matrix for average FC-values. The default is avg_corr.
    save : bool, optional
        decide, if plot should be saved as .PNG. The default is False.

    Returns
    -------
    None.

    '''
    # Prepare -----------------------------------------------------------------
    # Set Plotsize
    plt.figure(figsize=(15,12), dpi = 480)
    
    # Make Plot ---------------------------------------------------------------
    # Plot the heatmap with annotation
    heatmap = sns.heatmap(corr_mat, annot=True, fmt='.2f')
    
    # Get the color bar object
    cbar = heatmap.collections[0].colorbar
    
    # Set label and adjust its font size
    cbar.set_label(r'$M_r$', fontsize=25)
    
    # Save the plot
    if save:
        plt.savefig('plots//corr.png')
    
    # Show Plot
    plt.show()
    
    # Return None -------------------------------------------------------------
    return
    
plot_fc(save = True)

### FC means for each predictor group
fc_df = fn.wrap_up_connectivity('df')
fc_df = fc_df.drop(columns = ['ID'])

# Different calculation method. 
def get_avg_fc (data = fc_df, absolute = False):
    '''
    Function to compute average FC strength for between and within networks. 
    Besides default average also absolute average can be computed.

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
    # Preparation -------------------------------------------------------------
    # Prepare dictionary
    avg_fc = {'network':[], 'mean':[], 'sd':[]}
    
    # Get predictor names for within-connections
    predictor_groups = {}
    txt_files = ['reward.txt', 'self_reference.txt', 'attention.txt']
    
    # Read text files with assigned FCs for each network
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
    
    # Value Assignment --------------------------------------------------------
    # Average over total, inner and outer
    for conn in columns.keys():
        avg_fc['network'].append(conn)
        avg_fc['mean'].append(np.mean(data[columns[conn]]))
        avg_fc['sd'].append(np.nanstd(data[columns[conn]])) 
    
    # Average over networks
    for group in predictor_groups.keys():
        avg_fc['network'].append(group)
        avg_fc['mean'].append(np.mean(data[list(predictor_groups[group])]))
        avg_fc['sd'].append(np.nanstd(data[list(predictor_groups[group])]))
        
    # Return DataFrame with M/SD for each group--------------------------------    
    return pd.DataFrame(avg_fc)

avg_fc = get_avg_fc()
abs_avg_fc = get_avg_fc(absolute=True)

print('The average FC is:')
print(avg_fc)
print('\n The absolute average FC is:')
print(abs_avg_fc)

############################ ML Results #######################################
# Get Results Data
path_to_results = os.path.join(os.getcwd(),'res_ml_IAT_CV\IAT_results.pickle')
res = pd.read_pickle(path_to_results)
# Get ML Data
path_to_data = os.path.join(os.getcwd(),'data\ml_data_prep.xlsx')
if os.path.exists(path_to_data):
    data = pd.read_excel(path_to_data)
else:
    data = fn.get_prepared_data()

# Get mae, mse and r2 scores
def get_scores(res = res):
    '''
    Function to extract performance scores of the model

    Parameters
    ----------
    res : Dictionary, optional
        Dictionary including each metric for each CV-iteration (100).
        The default is res.

    Returns
    -------
    results : DataFrame
        Metrics as columns and CV-iterations as rows.
    summary : DataFrame
        Summary across all iterations for each metric.

    '''
    # Preparation -------------------------------------------------------------
    # prepare dictionary to save results in
    scores_dict = {'original':[],'shuffled':[]}
    
    # Extract Performance Metrics ---------------------------------------------
    # iterate over versions and metrics
    for version in ['original','shuffled']:
        # initialize metrics
        metrics = ['mae','mse','r2']
        # add shuffle indicator
        if version == 'shuffled':
            ind = '_sh'
        else:
            ind = ''
        # initialize score list
        tot_scores = []
        
        # iterate through CV-iterations
        for iteration, result in enumerate(res['scores'+ind]):
            iter_scores = []
            # Extract scores
            for score in metrics:
                iter_scores.append(result[score])
            tot_scores.append(iter_scores)
        
        # save as Dataframe
        df         = pd.DataFrame(tot_scores)
        metrics    = [m+ind for m in metrics]
        df.columns = metrics
        
        # save in dictionary
        scores_dict[version] = df
        
    # merge in one DataFrame
    results = pd.merge(scores_dict['original'], scores_dict['shuffled'],
                       left_index=True, right_index=True)
        
    # Extract Hyperparameters -------------------------------------------------
    ## prepare lists to save in
    colsample_bytree = []
    extra_trees      = []
    path_smooth      = []
    
    # iterate through CV-iterations
    for iteration in range(len(res['best_params'])):
        colsample_bytree.append(res['best_params'][iteration]['estimator__regressor__colsample_bytree'])
        extra_trees.append(res['best_params'][iteration]['estimator__regressor__extra_trees'])
        path_smooth.append(res['best_params'][iteration]['estimator__regressor__path_smooth'])
    
    # save in dataframe
    results['colsample_bytree'] = colsample_bytree
    results['extra_trees']      = extra_trees
    results['path_smooth']      = path_smooth

    summary = results.describe()
    
    # Return Results and Summary of Results -----------------------------------
    return results, summary

results, summary = get_scores()
description = results.describe()

############################### Model Scores ##################################
# Perform Linear Regressions on Scores by Hyperparameters
## Prepare columns
results['extra_trees'] = results['extra_trees'].astype(int)

## Test for normality of 'Path Smooth'
shapiro_test = stats.shapiro(results['path_smooth'])
print('Shapiro-Wilk test statistic:', shapiro_test[0])
print('Shapiro-Wilk test p-value:', shapiro_test[1])

# Fit Model
X = sm.add_constant(results[['path_smooth', 'colsample_bytree']])
model = sm.OLS(results['mae'], X).fit()
print(model.summary())

## Residuals
residuals = model.resid

## Shapiro-Wilk test for normality
shapiro_test = stats.shapiro(residuals)
print('Shapiro-Wilk Test Results:')
print('Test Statistic:', shapiro_test[1])
print('p-value:', shapiro_test[1])

## QQ-Plot for Normality
qq_plot()

## Breusch-Pagan test for homoscedasticity 
bp_test = sm.stats.diagnostic.het_breuschpagan(residuals, X)

## Extract test statistics and p-values
test_statistic = bp_test[0]
p_value = bp_test[1]

## Display results
print('Breusch-Pagan Test Results:')
print('Test Statistic:', test_statistic)
print('p-value:', p_value)

# Compute t-tests for model evaluation
## Identify Outliers
def winsorising(win_data):
    '''
    Function to perform winsorising: outliers which are revealed by 1.5xIQR rule
    are assigned new values such that they stay maximum values but do not bias
    the mean as much.

    Parameters
    ----------
    data : pd.Series or list
        Including values which schould be winsorized. If input is DataFrame 
        Winsorising is applied columnwise

    Returns
    -------
    data : pd.Series
        Transformed pd.Series in the same shape as input.

    '''
    # Preparation -------------------------------------------------------------
    # Check for data type
    if isinstance(win_data, list):
        win_data = pd.Series(win_data)
        is_list  = True
    else:
        is_list  = False
    
    # Create a copy
    win_data = win_data.copy()
    
    # Transformation ----------------------------------------------------------
    # Calculate quartiles and IQR
    Q1  = np.percentile(win_data, 25)
    Q3  = np.percentile(win_data, 75)
    IQR = Q3 - Q1
    
    # Identify outliers
    lower_bound      = Q1 - 1.5 * IQR
    upper_bound      = Q3 + 1.5 * IQR
    outliers_indices = np.where((win_data < lower_bound) | (win_data> upper_bound))
    outliers_list    = outliers_indices[0].tolist()
    
    # Assign new values
    win_data[outliers_list] = np.clip(win_data[outliers_list],
                                      lower_bound,
                                      upper_bound)
    
    # Return Transformed Data -------------------------------------------------
    # If necessary chanve return type
    if is_list:
        win_data = list(win_data)
    # Return
    return win_data

# Conduct t-tests to assess the difference in model performance between versions
def score_t_statistics(data = results, win = False):
    '''
    Function to perform paired one-sample ttests on model perfomance scores
    based on model version. Furthermore effect size and descriptive statistics
    are computed and stored

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame which is obtained by the get_scores function. All performance
        scores in the respective CV-iteration are contained.
        The default is results
    win : bool, optional
        Decide if data should be winsorized before analysis. The decision has 
        to be done after inspecting box-plots for each metric.
        The default is False.

    Returns
    -------
    score_stats : Dictionary
        Sorted after each metric, all important stats for ttest (mean, sd
        t-value, p-value, CI and d) are stored.

    '''
    # Preparation -------------------------------------------------------------
    # Objects to store results
    score_names = ['mae', 'mse', 'r2']
    score_stats = {}
    
    # Check if data should be winsorized
    if win == True:
        for col in data.columns:
            if data[col].dtype != bool:
                data[col] = winsorising(data[col].copy())
    
    # Analysis ----------------------------------------------------------------
    # Conduct t-tests for the pre-specified performance scores
    for i in score_names: 
        # Tested direction differs between r2 and the other two metrics
        if i == 'r2':
            d = (st.mean(data[i])-
                 st.mean(data[i+'_sh']))/st.stdev(data[i+'_sh'])
            stat_res = pg.ttest(x=data[i],
                                y=data[i+'_sh'],
                                paired=True,
                                alternative = 'greater')
        else:
            d = (st.mean(data[i+'_sh'])-
                 st.mean(data[i]))/st.stdev(data[i+'_sh'])
            stat_res = pg.ttest(x=data[i],
                                y=data[i+'_sh'],
                                paired=True,
                                alternative = 'less')
            
        # Get means and sd's of groups
        mean = np.mean(data[i])
        mean_sh = np.mean(data[i+'_sh'])
        sd = np.std(data[i])
        sd_sh = np.std(data[i+'_sh'])
        
        # Store values in dictionary
        score_stats[i] = {'t_statistic' : stat_res.iloc[0,0],
                          'p_value' : stat_res.iloc[0,3],
                          'd' : d,
                          'mean':mean,
                          'mean_sh':mean_sh,
                          'sd':sd,
                          'sd_sh':sd_sh,
                          'CI95%':stat_res.iloc[0,4]}
    
    # Return Results of Analysis ----------------------------------------------
    return score_stats

score_stats = score_t_statistics(win = False)
score_stats_win = score_t_statistics(win = True)

# Convert to usual decimal form as a string
def get_normal_notation(x):
    normal = '{:.15f}'.format(x)
    return normal
print(f'Comparison of mean-absolute-error (MAE) results in ' 
      f't:{round(score_stats_win["mae"]["t_statistic"],4)}, '
      f'p:{get_normal_notation(score_stats_win["mae"]["p_value"])[0:5]} ' 
      f'and d:{round(score_stats_win["mae"]["d"],3)}')
print(f'Comparison of mean-squared-error (MSE) results in ' 
      f't:{round(score_stats_win["mse"]["t_statistic"],4)}, '
      f'p:{get_normal_notation(score_stats_win["mse"]["p_value"])[0:5]} ' 
      f'and d:{round(score_stats_win["mse"]["d"],3)}')
print(f'Comparison of r-squared (r2) results in ' 
      f't:{round(score_stats_win["r2"]["t_statistic"],4)}, '
      f'p:{get_normal_notation(score_stats["r2"]["p_value"])[0:5]} ' 
      f'and d:{round(score_stats_win["r2"]["d"],3)}')

######################### SHAP-scores #########################################
## Scores
def get_shap(res = res, shuffled = False):
    '''
    Function to extract the SHAP-scores out of the results object

    Parameters
    ----------
    res : dictionary, optional
        Dictionary with all results where the SHAP-scores need to be extracted
        out. The default is res.
    shuffled : bool, optional
        Decision if the SHAP-scores from the original or the shuffled model 
        should be extracted. The default is False.

    Returns
    -------
    shap_scores : Dictionary
        Ditionary containing dataframes for each CV-iteration. The dataframes
        consist of SHAP-scores for each subject in the respective test-set (rows)
        and each predictor (column)
    shap_score_summary : Dictionary
        Summary fopr each DataFrame in shap_scores.
    shap_score_mean : Dictionary
        Keys for each predictor in which a list with all mean SHAP-scores for
        each iteration is stored.

    '''
    # Preparation -------------------------------------------------------------
    # Name the version where to extract the values
    if shuffled:
        expl = 'explainations_sh'
    else:
        expl = 'explainations'
        
    # Prepare storing objects    
    shap_scores = {}
    shap_score_summary = {}
    
    # Extract SHAP-scores -----------------------------------------------------
    # Iterate through each CV-iteation
    for i, j in enumerate(res[expl]):
        # Extract SHAP-scores
        df = pd.DataFrame(j.values, columns=j.data.columns)
        shap_scores[i] = df
        # Get description for each extracted DataFrame
        df_desc = pd.DataFrame(df.describe())
        shap_score_summary[i] = df_desc
    
    # Get average SHAP-score for each predictor and iteration 
    columns = shap_score_summary[0].columns
    shap_score_mean = {}
    for col in columns:
        l = []
        for iteration in range(len(shap_score_summary)):
            l.append(shap_score_summary[iteration][col][1])
        shap_score_mean[col] = l
        
    # Return SHAP-scores in differen objects ----------------------------------    
    return shap_scores, shap_score_summary, shap_score_mean

shap_scores, shap_score_summary, shap_score_mean = get_shap(shuffled = False)
shap_scores_sh, shap_score_summary_sh, shap_score_mean_sh = get_shap(shuffled = True)
    
# Get Dataframes for Insights
def get_shap_descriptions(shap_score_mean = shap_score_mean, 
                          shap_score_mean_sh = shap_score_mean_sh):
    '''
    Function to compute the descriptions for SHAP-scores of both versions.

    Parameters
    ----------
    shap_score_mean : Dictionary, optional
        Features as keys containing lists of SHAP-values for each CV-iteration.
        The default is shap_score_mean.
    shap_score_mean_sh : Dictionary, optional
        Features as keys containing lists of SHAP-values for each CV-iteration.
        The default is shap_score_mean_sh.

    Returns
    -------
    shap_description : DataFrame 
        Description of SHAP values for the original model.
    shap_description_sh : DataFrame 
        Description of SHAP values for the shuffled model.

    '''
    # Preparation -------------------------------------------------------------
    # Turn dictionaries in DataFrames
    shap_dataframe = pd.DataFrame(shap_score_mean)
    shap_dataframe_sh = pd.DataFrame(shap_score_mean_sh)
    
    # Compute Description -----------------------------------------------------
    # Get description of DataFrames
    shap_description = pd.DataFrame(shap_dataframe.describe())
    shap_description_sh = pd.DataFrame(shap_dataframe_sh.describe())
    
    # Return Description ------------------------------------------------------
    return shap_description, shap_description_sh

shap_description, shap_description_sh = get_shap_descriptions()

# Prepare SHAP values for t.tests
def shap_t_statistics(shap_score_mean = shap_score_mean, 
                      shap_score_mean_sh = shap_score_mean_sh,
                      win = False):
    '''
    Function to conduct t-tests for SHAP-scores of each predictor between both
    model versions
    
    Parameters
    ----------
    shap_score_mean : Dictionary, optional
        Features as keys containing lists of SHAP-values for each CV-iteration.
        The default is shap_score_mean.
    shap_score_mean_sh : Dictionary, optional
        Features as keys containing lists of SHAP-values for each CV-iteration.
        The default is shap_score_mean_sh.
    win : bool, optional
        Decide if data should be winsorized before analysis. The decision has 
        to be done after inspecting box-plots for each metric.
        The default is False.

    Returns
    -------
    comparison : dictionary
        Features as key containing a DataFrame each with columns 'original' and
        'shuffled' to compare SHAP-values.
    shap_statistics : dictionary
        Statistics for t-test as key containing dicitionaries with feautes as
        keys and respective statistics as array.

    '''
    # Preparation -------------------------------------------------------------
    # Create empty dictionary
    comparison = {}
    # Turn data into DataFrame to conduct t-tests on
    for i in list(shap_score_mean.keys()):
        comparison[i] = pd.DataFrame({'real':shap_score_mean[i],
                                  'shuffled':shap_score_mean_sh[i]})
        
    # Perform winsorising if needed
    if win:
        for feat in comparison.keys():
            for col in comparison[feat].columns:
                comparison[feat][col] = winsorising(comparison[feat][col])
    
    # Create empty dictionaries for statistics
    t_stat = {}
    p_val = {}
    confint = {}
    d = {}
    mean = {}
    mean_sh = {}
    sd = {}
    sd_sh = {}
    
    # Computation -------------------------------------------------------------
    # Compute t-tests for each feature
    for i in list(comparison.keys()):
        # Conduct t-test
        stat_res = pg.ttest(x=comparison[i]['real'], 
                            y=comparison[i]['shuffled'], 
                            paired=True, 
                            alternative = 'greater')
        
        # Compute and store relevant statistics
        t_stat[i] = stat_res.iloc[0,0]
        p_val[i] = stat_res.iloc[0,3]
        confint[i] = stat_res.iloc[0,4]
        d[i] = stat_res.iloc[0,5]
        mean[i] = np.mean(shap_score_mean[i])
        mean_sh[i] = np.mean(shap_score_mean_sh[i])
        sd[i] = np.std(shap_score_mean[i])
        sd_sh[i] = np.std(shap_score_mean_sh[i])
    
    # Save summary in dictionary
    shap_statistics = {'t_statistic':t_stat, 'p_value': p_val,
                       'CI95%':confint, 'd':d, 'mean':mean,
                       'mean_sh':mean_sh, 'sd':sd, 'sd_sh':sd_sh}
    
    # Return Results ----------------------------------------------------------    
    return comparison, shap_statistics

comparison, shap_statistics = shap_t_statistics()
comparison_win, shap_statistics_win = shap_t_statistics(win = True)

# List absolute mean SHAP values
def list_overall_mean(shap_score_mean = shap_score_mean,
                      shap_score_mean_sh = shap_score_mean_sh):
    '''
    Function to store absolute mean SHAP-values for each feature to get descriptive 
    overview about the strongest features

    Parameters
    ----------
    shap_score_mean : Dictionary, optional
        Features as keys containing lists of SHAP-values for each CV-iteration.
        The default is shap_score_mean.
    shap_score_mean_sh : Dictionary, optional
        Features as keys containing lists of SHAP-values for each CV-iteration.
        The default is shap_score_mean_sh.

    Returns
    -------
    overall_mean : Dictionary
        Means of SHAP-values for features for both model versions.
    overall_sd : Dictionary
        Standard deviations of SHAP-values for features for both model versions.

    '''
    # Preparation -------------------------------------------------------------
    # Split data in both model versions 
    data = {'original':shap_score_mean, 'shuffled':shap_score_mean_sh}
    
    # Create dictionaries to store results
    absolute_mean = {'original':{}, 'shuffled':{}}
    absolute_sd = {'original':{}, 'shuffled':{}}
    
    # Compute Metrics ---------------------------------------------------------
    # Iterate over each model version
    for version in data.keys():
        # Iterate over each predictor
        for predictor in shap_score_mean.keys():
            # Get Averages and SDs for each predictor
            absolute_mean[version][predictor] = np.mean(data[version][predictor])
            absolute_sd[version][predictor] = np.std(data[version][predictor])
              
    # Return Computed Metrics -------------------------------------------------                                                 
    return absolute_mean, absolute_sd

listed_predictors, listed_predictors_sd = list_overall_mean()

# Get min, max and sd of all SHAP scores for every predictor
def conn_stats(scores = shap_scores):
    '''
    Function to compute stats (mean, min, max, sd, var) of SHAP-values for each
    feature to enable descriptive listing

    Parameters
    ----------
    scores : TYPE, optional
        DESCRIPTION. The default is shap_scores.

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    '''
    # Preparation -------------------------------------------------------------
    # Set up object to store results in
    result = {'connectivity':[],'min':[],'max':[],'sd':[], 'var':[], 'mean':[]}
    conn_dict = {}
    
    # Computation -------------------------------------------------------------
    # Iterate over each CV-iteration and respective SHAP-dataframe
    for df_name, df in scores.items():
        for column in df.columns:
            # If the column name is not already a key in combined_data, initialize it as an empty list
            if column not in conn_dict:
                conn_dict[column] = []
            # Append the column values to the corresponding key in combined_data
            conn_dict[column].extend(df[column].tolist())
    
    # Storing -----------------------------------------------------------------
    # Store computed values in prepared object
    for key in conn_dict.keys():
        result['connectivity'].append(key) 
        result['min'].append(min(conn_dict[key])) 
        result['max'].append(max(conn_dict[key])) 
        result['sd'].append(np.std(conn_dict[key]))
        result['var'].append(np.var(conn_dict[key]))
        result['mean'].append(np.mean(conn_dict[key]))
            
    # Return ------------------------------------------------------------------
    # Turn result in DataFrame
    result = pd.DataFrame(result)
    
    # Return
    return result

shap_desc = {'original':conn_stats(shap_scores),
             'shuffled':conn_stats(shap_scores_sh)}

# Prepare Hypothesis Testing
def get_group_means(scores = shap_scores, scores_sh = shap_scores_sh):
    '''
    Function to compute the average SHAP-value for each predictor group (network)
    to enable statistical comparison (t-test) between original and shuffled model. 

    Parameters
    ----------
    scores : Dictionary, optional
        CV-Iterations as keys, containing DataFrames with (original) 
        SHAP-values for each instance and feature.
        The default is shap_scores.
    scores : Dictionary, optional
        CV-Iterations as keys, containing DataFrames with (shuffled) 
        SHAP-values for each instance and feature.
        The default is shap_scores_sh.

    Returns
    -------
    result : Dictionary
        Containg two dictionaries (original/shuffled) each containing another
        dictionary for each network and respective SHAP-values for each 
        CV-Iteration.

    '''
    
    # Preparation -------------------------------------------------------------
    # Empty dictionaries to store values
    predictor_groups = {}
    
    result = {'original': {'reward':[],'self_reference':[],'attention':[]},
              'shuffled': {'reward':[],'self_reference':[],'attention':[]}}
    
    # Combine Input
    versions = {'original':scores, 'shuffled':scores_sh}
    
    # Specify Textfiles for retrieval of associated features for each network
    txt_files = ['reward.txt', 'self_reference.txt', 'attention.txt']
    
    # Iterate through textfiles, open them and store associated features
    for txt_file in txt_files:
        file_path = os.path.join('data',txt_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            group_name = os.path.splitext(os.path.basename(txt_file))[0]
            predictor_groups[group_name] = [column.strip() for column in lines]

    # Computation -------------------------------------------------------------
    # Iterate over versions (original/shuffled)
    for v in versions.keys():
        # Subset scores
        scores = versions[v]
        # Prepare dictionary for averages of sums
        averages = {'reward':[], 'self_reference':[], 'attention':[]}
        
        # Iterate over each CV-iteration and respective SHAP-dataframe
        for df_name, df in scores.items():
            # Create an empty DataFrame to store results
            sums = {'reward':[], 'self_reference':[], 'attention':[]}
            # Iterate through each predictor group
            for group_name in predictor_groups:
                # Iterate through each instance
                for row_index, row in df.iterrows():
                    # Compute sum
                    sum_value = sum(row[predictor_groups[group_name]])
                    sums[group_name].append(sum_value)
                # Compute average over for each network
                avg_value = np.mean(sums[group_name])
                averages[group_name].append(avg_value)
        
        # Store computations
        for i in result[v].keys():
            result[v][i] = averages[i]
                
    # Return ------------------------------------------------------------------
    return result

network_results = get_group_means()

# Conduct Hypothesis Testing
def network_t_tests(network_results = network_results,
                    win = False):
    '''
    Function to perform hypothesis testing in conducting t-tests on SHAP-value
    differences between both model versions for each network

    Parameters
    ----------
    network_results : Dictionary, optional
    Containg two dictionaries (original/shuffled) each containing another
    dictionary for each network and respective SHAP-values for each
    CV-Iteration.. The default is network_results.
    
    win : bool, optional
        Decision if winsorising should be applied to input data.
        The default is False.

    Returns
    -------
    hypothesis_tests : Dataframe
        Rows as networks and columns with relevant t-test statistics.

    '''
    
    # Preparation--------------------------------------------------------------
    # Retrieve network names
    networks = list(network_results['original'].keys())
    
    # Set up dictionary to store results in
    hypothesis_tests = {'network':[],
                        't_statistic':[], 
                        'p_value':[],
                        'CI95%':[],
                        'd':[], 
                        'df':[],
                        'mean':[],
                        'sd':[],
                        'mean_sh':[],
                        'sd_sh':[]}
    
    # Check if winsorisation should be applied
    if win == True:
        for version in network_results.keys():
            for network in network_results['original'].keys():
                network_results[version][network] = winsorising(network_results[version][network].copy())
    
    # Analysis-----------------------------------------------------------------
    # Iterate thorugh each network
    for network in networks:
        # Perform t-test
        stat_res = pg.ttest(x=network_results['original'][network],
                            y=network_results['shuffled'][network],
                            paired=True, alternative = 'greater')
        
        # Get network means and sd for each version
        mean = np.mean(network_results['original'][network])
        sd = np.std(network_results['original'][network])
        mean_sh = np.mean(network_results['shuffled'][network])
        sd_sh = np.std(network_results['shuffled'][network])
        
        # Prepare Storing of results in prepared dictionary
        stats_stored = [network, 
                        stat_res.iloc[0,0],
                        stat_res.iloc[0,3],
                        stat_res.iloc[0,4],
                        stat_res.iloc[0,5], 
                        stat_res.iloc[0,2],
                        mean,
                        sd,
                        mean_sh,
                        sd_sh]
        
        # Store values in Dictionary
        for i, value in enumerate(hypothesis_tests.keys()):
            hypothesis_tests[value].append(stats_stored[i])
    
    # Turn results-dictionary into DataFrame
    hypothesis_tests = pd.DataFrame(hypothesis_tests)
    
    # Return-------------------------------------------------------------------
    # Return DataFrame
    return hypothesis_tests

hypothesis_tests = network_t_tests()
hypothesis_tests_win = network_t_tests(win = True)

# Compute deviations of p-value through Winsorizing
# Prepare ---------------------------------------------------------------------
# Set up list for storing
p_dev = []

# Compute ---------------------------------------------------------------------
# Iterate through each model score
for score in ['mae','mse','r2']:
    p_dev.append(score_stats_win[score]['p_value']-score_stats[score]['p_value'])
    
# Add winsorisation deviations from hypothesis tests
p_dev = p_dev + list(hypothesis_tests_win['p_value']-hypothesis_tests['p_value'])

# Add winsorisation deviations from exploratory analysis
p_dev.append(shap_statistics_win['p_value']['age']-shap_statistics['p_value']['age'])
p_dev.append(shap_statistics_win['p_value']['gender']-shap_statistics['p_value']['gender'])

# Print Results ---------------------------------------------------------------
print('Average p_value deviation is:', round(np.mean(p_dev),3), 
      'and respective SD is:', round(np.std(p_dev),3))


###################################### Plots ##################################

# Plot Model Performance evaluation
def plot_model_scores(data = results,
                      save = False):
    '''
    Function to plot comparison boxplots for t-tests used for model performance
    evaluation.

    Parameters
    ----------
    data : DataFrame, optional
        Metrics as columns and CV-iterations as rows.
        The default is results.
    save : bool, optional
        Decide, if plot should be saved.
        The default is False.

    Returns
    -------
    None.

    '''
    # Preparation -------------------------------------------------------------
    # Specify which scores are paired together
    pairs = [('mae', 'mae_sh'),
             ('mse', 'mse_sh'),
             ('r2', 'r2_sh')]
    
    # Label common names
    labels = ['MAE', 'MSE', 'r2']
    
    # Plot --------------------------------------------------------------------
    # Set Theme
    sns.set_theme(style = 'whitegrid')
    
    # Create a 1x3 subplot grid for three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    
    # Loop through the pairs and create boxplots in each subplot
    for i, (col1, col2) in enumerate(pairs):
        ax = axes[i]
        # Make Plot
        sns.boxplot(data=data[[col1, col2]], 
                    ax=ax, 
                    palette = sns.color_palette('Paired'), 
                    boxprops=dict(alpha=1))
        # Add correct r-squared writing
        if labels[i] == 'r2':
            label = r'$R^2$'
        else: 
            label = labels[i]
            
        # Customize Plot 
        # Set the x-axis label as the name of the first column
        ax.set_xlabel(label, 
                      fontsize = 14)
        # Set x-axis tick labels to 'Real' and 'Shuffled'
        ax.set_xticklabels(['Original', 'Shuffled'],
                           fontsize = 14)
        # Add Datapoints
        ax = sns.stripplot(data=data[[col1, col2]], 
                           ax=ax, color="orange", 
                           jitter=0.1, 
                           size=4.5)
    
    # Adjust layout
    plt.tight_layout()
    # Save Plot
    if save:
        plt.savefig('plots//model_scores_plot.png', dpi = 300)
    # Show Plot
    plt.show()
    
    # Return None -------------------------------------------------------------
    return None

plot_model_scores()

# Plot Hypothesis Testing
def plot_hypothesis_testing(network_results = network_results,
                            save = False):
    '''
    Function to plot comparison boxplots for t-tests used for hypothesis 
    testing.

    Parameters
    ----------
    network_results : Dictionary, optional
    Containg two dictionaries (original/shuffled) each containing another
    dictionary for each network and respective SHAP-values for each
    CV-Iteration.. The default is network_results.
    save : bool, optional
        Decision if plot should be saved.
        The default is False.

    Returns
    -------
    None.

    '''
    # Preparation -------------------------------------------------------------
    # Bring relevant data in DataFrame
    data = pd.DataFrame({'Reward':network_results['original']['reward'],
                         'Reward_SH':network_results['shuffled']['reward'],
                         'Self-Reference':network_results['original']['self_reference'],
                         'Self-Reference_SH':network_results['shuffled']['self_reference'],
                         'Attention':network_results['original']['attention'],
                         'Attention_SH':network_results['shuffled']['attention']})
    
    # Turn DataFrame into long format
    data_long = pd.melt(data,
                        var_name='Network',
                        value_name='SHAP-Value')
    data_long['Version'] = data_long['Network'].str.endswith('_SH').map({True: 'Shuffled', False: 'Original'})
    data_long['Network'] = data_long['Network'].str.replace('_SH', '')
    
    # Plot --------------------------------------------------------------------
    # Set Theme
    sns.set_theme(style='whitegrid')
    
    # Set the figure size
    fig, ax = plt.subplots(figsize=(10, 6))  
    
    # Make Boxplot
    sns.boxplot(data=data_long, 
                x='Network', 
                y='SHAP-Value', 
                hue='Version', 
                palette = [sns.color_palette('Paired')[0],
                           sns.color_palette('Paired')[1]],
                ax=ax)
    
    # Add Datapoints
    sns.stripplot(data=data_long, 
                  x='Network', 
                  y='SHAP-Value',
                  dodge=True,
                  hue='Version', 
                  palette=[sns.color_palette("Paired", 8)[6],'orange'],
                  alpha=0.5,
                  ax=ax)
    
    # Remove the legend
    ax.legend().set_visible(False)
    
    # Customize axis labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Reward', 'Self-Reference','Attention'], fontsize = 14)
    ax.set_ylabel("SHAP-Value", fontsize = 14)
    ax.set_xlabel("")
    
    # Save Plot
    if save :
        plt.savefig('plots//hypothesis_testing2_plot.png', dpi=300)
    
    # Show Plot
    plt.show()
    
    # Return None -------------------------------------------------------------
    return None

# Call the function with your DataFrame
plot_hypothesis_testing()
