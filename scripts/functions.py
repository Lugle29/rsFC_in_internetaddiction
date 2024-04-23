"""
@title: functions
@author: l.glenzer
"""

#----------------Packages------------------------------------------------------
import requests
import zipfile as zp
import os
import scipy.io
import pandas as pd
import re
from itertools import compress

#----------------Functions-----------------------------------------------------

#----------------Behavioral Data-----------------------------------------------

def download_behavioral_data(
        url:str = "https://www.nitrc.org/frs/download.php/10910/behavioral_data_MPILMBB_v3.zip//?i_agree=1&download_now=1", 
        zip_path:str = "data_behav.zip", 
        dir_path:str = os.path.join("data", "behavioral"), 
        target_path:str = "behavioral_data_MPILMBB"):
    '''
    Function to download the behavioral data from URL provided in paper

    Parameters
    ----------
    url : str, optional
        URL from paper plus commands to download. 
        The default is "https://www.nitrc.org/frs/download.php/10910/behavioral_data_MPILMBB_v3.zip//?i_agree=1&download_now=1".
    zip_path : str, optional
        DESCRIPTION. The default is "data_behav.zip".
    dir_path : str, optional
        DESCRIPTION. The default is os.path.join("data", "behavioral").
    target_path : str, optional
        DESCRIPTION. The default is "behavioral_data_MPILMBB".

    Returns
    -------
    None.

    '''
    zip_path = os.path.join(dir_path, zip_path)
    target_path = os.path.join(dir_path, target_path)
    
    if os.path.exists(target_path):
        print("Behavioral data was already donwloaded and unzipped and is therefore ready to use.")
    else:
        if os.path.exists(zip_path):
            print("Zip-file was already downloaded and will be unzipped subsequently.")
            with zp.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dir_path)
            print("Behavioral data is now ready to use.")
        else:
            print("Behavioral data will be downloaded and unzipped subsequently.")
            response = requests.get(url)
            with open(zip_path, "wb") as f:
                f.write(response.content)
            print("Download was successful.")    
            with zp.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dir_path)
            print("Behavioral data is now ready to use.")
    return()


# Retrieve files and prepare data in list
def prepare_data(filedir:str, type_of_file:str = ".tsv"):
    '''
    .tsv files will be read and stored as single dataframes in a dictionary

    Parameters
    ----------
    filedir : str
        Directory where data can be found.
    type_of_file : str, optional
        Which kind of file should be read. The default is ".tsv".

    Returns
    -------
    filelist : TYPE
        List of names of behavioral measurements.
    data_dict : TYPE
        dictionary containing df of the behavioral measurements.

    '''
    # Creating list with all .tsv files
    filelist = []
    print("A list with all ",type_of_file," files in the chosen directory is created\n")
    for f in os.listdir(filedir):
        if f.endswith(type_of_file):
            filelist.append(f)
    print("Filelist ready\n")
    # Save data in list
    print("Filelist will be converted into dictionary\n")
    data_dict = {}
    while True:
        if (type_of_file == ".tsv"):
            for i in filelist:
                data_dict[i] = pd.read_csv(os.path.join(filedir, i), sep='\t', header=0)
            print("Dictionary is ready to further processing\n")
            break;
        else:
            print("Cannot process this kind of file. Use .tsv\n")
            break;
    return filelist, data_dict

    
# Prepare DataFrame
def ids(dataframe):
    '''
    Function to clean the ID column by removing 'sub-' for each value and rename the column to 'ids' for merging

    Parameters
    ----------
    dataframe : DataFrame
        Dataframe for each collected instrument.

    Returns
    -------
    dataframe : DataFrame
        Cleaned Dataframe for each collected instrument, such that the ID-column is called 'ids' and values are 5-digit numbers.

    '''
    pd.options.mode.chained_assignment = None  # default='warn'
    
    for i in range(len(dataframe.iloc[:, 0])):
        dataframe.iloc[:, 0][i] = re.sub('sub-', '', dataframe.iloc[:, 0][i])
        
    dataframe.rename(columns={'participant_id': 'ids'}, inplace=True)
    dataframe['ids'] = dataframe['ids'].astype(int)
    
    return (dataframe)

# Function for merging dataframes
def merge_instruments(directory:str, filelist:list, startingdf:int=0):
    '''
    All collected measures will be merged into one DataFrame with a single 'ids' column and instruments as columns
    
    Parameters
    ----------
    directory : str
        Sets the directory where downloaded data is to be found.
    filelist : list
        Includes the files which are contained by dataset.
    startingdf : int, optional
        Sets the dataframe the iteration starts with. The default is 0.

    Returns
    -------
    df : dataframe
        Containing all behavioral measures with default names.
    '''
    
    df = ids(pd.DataFrame(directory[filelist[startingdf]]))
    for i in filelist:
        if i != filelist[startingdf]:
            df2 = pd.DataFrame(directory[i])
            if df2.columns[0] != 'ids':
                df2 = ids(df2)
            df = pd.merge(df, df2, on='ids', how='outer')
    return df

# Clean behavioral Data
def behav_to_work(
        df, 
        cols = ['ids','IAT_sum'],
        col_names = ['ID','IAT', 'fMRIavailable'],
        ID_to_index = False):
    '''
    Prepare the merged DataFrame such that only the relevant behavioral measures 
    are kept, and column names are changed. The transformed DataFrame is saved as .xlsx

    Parameters
    ----------
    df : DataFrame
        Merged DataFrame with all behavioral measures.
    cols : list, optional
        Specify which columns to keep. The default is ['ids','IAT_sum', 'fmri_availbl'].
    col_names : list, optional
        Provide names, how to call the kept columns. The default is ['ID','IAT','fMRI_available'].
    ID_to_index : bool, optional
        Specify if the ID column should be converted to the index. The default is False.

    Returns
    -------
    df_sub : DataFrame
        DataFrame with only the relevant measures and renamed.

    '''
    # Check if input is correct
    if type(cols) != list:
        cols = list(cols.split(" "))
    if type(col_names)!= list:
        col_names = list(col_names.split(" "))
    
    # Subset DataFrame
    df_sub = df[cols].copy()
    # Add Column informing about the availability of fMRI Data
    fmri_subs = open("data/fmri-subs.txt", "r").readlines()[0]
    df_sub["fmri_availble"] = [str(code) in fmri_subs for code in df["ids"]]
    # Rename Columns
    df_sub.columns = col_names
    # Save Transformed DataFrame
    df_sub.to_csv(os.path.join("data", col_names[1] + "behav_work_data.csv"), index=False)
    df_sub.to_excel(os.path.join("data", col_names[1]+"behav_work_data.xlsx"), index=False)
    
    return df_sub

# Read behavioral data out of saved file
def open_behav_data(variable_of_interest = 'IAT_sum', name_of_interest = 'IAT'):
    '''
    Checks if behav data is already stored and reads it if so. Otherwise it will
    be downloaded and read then.

    Returns
    -------
    df : dataframe
        Containing subset behavioral data and indication for which IDs fMRI data
        is available.

    '''
    path_raw = os.path.join(os.getcwd(), "data", "MBB_behavioral.csv")
    path_work = os.path.join(os.getcwd(), "data", name_of_interest+"behav_work_data.csv")
    path_down = os.path.join(os.getcwd(), "data", "Behavioral", "behavioral_data_MPILMBB", "phenotype")
    
    if os.path.exists(path_work):
        # Processed behavioral data is available and is read
        df = pd.read_csv(path_work)
        
    elif os.path.exists(path_raw):
        # not fully processed behavioral data is avaiable and subsequently will be further processed and stored accordingly
        
        # read raw data
        df = pd.read_csv(path_raw)
        # subset raw data
        df = behav_to_work(df,
                           cols = ['ids',variable_of_interest],
                           col_names = ['ID',name_of_interest,'fMRI_available'])
        # save subset data
        df.to_csv(path_work, index=False)
        
    elif os.path.exists(path_down):
        # files were downloaded but not processed at all
        
        # merge mesures
        filelist, directory = prepare_data(path_down)
        df = merge_instruments(directory, filelist)
        # save merged data
        df.to_csv(path_raw, index=False)
        # subset raw data
        df = behav_to_work(df,
                           cols = ['ids',variable_of_interest],
                           col_names = ['ID',name_of_interest,'fMRI_available'])
        # save subset data
        df.to_csv(path_work, index=False)
        
    else:
        # data was not downloaded yet
        download_behavioral_data()
        
        # merge mesures
        filelist, directory = prepare_data(path_down)
        df = merge_instruments(directory, filelist)
        # save merged data
        df.to_csv(path_raw, index=False)
        # subset raw data
        df = behav_to_work(df,
                           cols = ['ids',variable_of_interest],
                           col_names = ['ID',name_of_interest,'fMRI_available'])
        # save subset data
        df.to_csv(path_work, index=False)
        
    return df

# Read Sociodemographic data and store it in dataframe
def sociodem():
    """
    Function to read .tsv file for sociodemographics, store it as df and prepare for merging.
    DF will be saved as .csv

    Returns
    -------
    socio : dataframe with ID, gender and age as columns

    """
    path_retr = os.path.join(os.getcwd(), 'data/Behavioral/participants.tsv')
    path_check = os.path.join(os.getcwd(), 'data/sociodem.csv')
    
    if os.path.exists(path_check):
        # Processed behavioral data is available
        socio = pd.read_csv(path_check)
    else:
    # Get data
        socio = pd.read_csv(path_retr, sep='\t', header=0)
        
        # Prepare for merging
        socio.rename(columns={'participant_id': 'ID', 'age (5-year bins)': 'age'}, inplace=True)
        socio['ID'] = socio['ID'].str.replace('sub-0', '')
        
        # Save as .csv
        socio.to_csv(path_check, index=False)
    
    return socio

#----------------Behavioral Data END-------------------------------------------
#----------------Functional-Data-----------------------------------------------

# Function to get the subject IDs which were analyzed in the CONN-Toolbox
def extract_no_subjects(
        directory = os.path.join("data", "SBC_01")):
    """
    Function to extract list of subjects for which data exists in a specific 
    folder to be able to iterate over different data file for each subject

    Parameters
    ----------
    directory : str. 
        Path of directory where subject-files should be found. 
        The default is os.path.join("data", "SBC_01").

    Returns
    -------
    subjects : one dimensional array containing the numbers of each subject 
    with available data in the directory.

    """
    
    # extracting list of all files or directories in path
    content = os.listdir(directory)
    
    # boolean list of file names containing "Subject" in it.
    sub = ["Subject" in name for name in content]
    
    # subset content to the file names containing "Subject"
    subjects = list(compress(content, sub))
    
    # extract only the number in three-places format in sorted order
    subjects = sorted([re.findall("(?<=Subject)...", sbj) for sbj in subjects])
    
    # return subjects
    return subjects

# Rearrange FC matrices such that one DataFrame with each subject can be built
def mat_corr_to_row(sub_dict, corr):
    """
    Transform corr matrices per subject to dictionary row-a-like
    Parameters
    ----------
    sub_dict : dictionary
        indicating subject ID
    corr : DataFrame
        DataFrame containing correlation matrix of subject

    Returns
    -------
    dictionary with subject and network combinations as keys and id & corr's as values'

    """
    net_names = corr.columns
    
    for i in range(len(net_names)):
        for j in range(i+1, len(net_names)):
            tmp1 = net_names[i]
            tmp2 = net_names[j]
            sub_dict[tmp1 + "_" + tmp2] = [corr[tmp1][tmp2]]
    
    # Return
    return sub_dict

# Function to extract the IDs as found for fMRI data
def extract_general_ids(directory:str = "data", file:str = "fmri-subs.txt"):
    '''
    Function to see which general IDs have been analysed with functional data

    Parameters
    ----------
    directory : str, optional
        specifies where the file can be found. The default is "data".
    file : TYPE, optional
        fiel where the IDs are stored. The default is "fmri-subs.txt".

    Returns
    -------
    fmri_subs : list
        List containing IDs which are used for fMRI analysis.

    '''
    path = os.path.join(directory, file)
    # Prepare fMRI subject-list
    fmri_subs = open(path, "r").readlines()[0]
    fmri_subs = fmri_subs.split("', '")
    fmri_subs[0] = fmri_subs[0][-5:]
    fmri_subs = fmri_subs[:-1]
    fmri_subs.sort()
    
    return fmri_subs

# Function to extract FC matrix
def extract_corr_mat(
        subject,
        file_dir_inp = os.path.join("data", "SBC_01"),
        names_file = str('FC_names.txt')
        ):
    """
    Function to extract correlations between different networks for one subject
    at a time.

    Parameters
    ----------
    subject : character, indicating the number of which subject is to be. To be
    retrieved out of subjects list
    file_dir : path, where file of subject can be found.
    extracted, which needs to be of the format 001 or 023 for 1 and 23.
    network_name : Name of atlas which was used.
        DESCRIPTION. The default is 'Yeo07'.

    Returns
    -------
    dictionary with the actual extracted subject and dataframe containing the
    correlation matrix

    """
    
    file_name = "".join(["resultsROI_Subject", subject, "_Condition001.mat"])
    
    wd = os.getcwd()

    file_dir = os.path.join(wd, file_dir_inp)
    
    # Load matlab file
    mat = scipy.io.loadmat(os.path.join(file_dir, file_name))
    
    # Extract names of networks
    net_names = mat['names']
    net_names = [label[0].tolist() for label in net_names[0]]
    
    if not os.path.exists(os.path.join(os.getcwd(), 'data/FC_names.txt')):
        names_file = os.path.join('data', names_file)
        with open(names_file, 'w') as file:
            for item in net_names:
                file.write(str(item) + '\n')
        
    # Extract correlation matrix
    corr = pd.DataFrame(mat['Z'], columns=net_names, index=net_names)
    
    # extract unique values with paired networks names
    res_dict = {"ID": [subject]}
    
    # Return
    return(res_dict, corr)

# Function to obtain merged connectivity data
def wrap_up_connectivity(
        output: str,
        conn_name = 'SBC_01',
        ID_to_index=False):
    '''
    Wraps up extracting all avaibale subjects and extracting connectivity data 
    of .mat-files to return a data frame with all avaliable subjects

    Parameters
    ----------
    output : str
        Specify desired output, to choose between DataFrame and Dictionary.
    conn_name : str
        Name of the Analysis in the Conn-Toolbox. The default is 'SBC_01'.
    ID_to_index : bool, optional
        Decide if the ID column should be converted as Index. The default is False.

    Returns
    -------
    Either DataFrame or Dictionary with ID identifier and respective connectivity values
    '''
    
    # Get relevant path
    file_dir_inp = os.path.join("data", conn_name)
    
    # possible output check: data frame or dictionary
    psbl_op = ["df","dict"]
    if output not in psbl_op:
        raise ValueError("Invalid value for 'ouput'. Must be either 'df' or 'dict'.")
    
    # set up
    subjects = extract_no_subjects()
    ids = extract_general_ids()[0:len(subjects)]
    
    if output == "df":
        # initialize output data frame
        df = pd.DataFrame()
        # get connectivity for each subject and merge to one DataFrame across subjects
        for sub in subjects:
            tmp_id, tmp_cor = extract_corr_mat(*sub, 
                                               file_dir_inp=file_dir_inp)
            tmp = pd.DataFrame(mat_corr_to_row(tmp_id, tmp_cor))
            df = pd.concat([df, tmp])
        
        # add ID column
        df["ID"] = [int(x) for x in ids]
        if ID_to_index:
            df = df.astype({'ID':'int'})
            df = df.set_index(df['ID'])
            df = df.drop(['ID'], axis = 1)
        return df
    
    elif output == "dict":
        # initialize output dictionary
        dictionary = {'ID':[], 'DF':[]}
        # get connectivity for each subject and merge to one DataFrame across subjects
        for sub in subjects:
                ID, corr = extract_corr_mat(*sub, 
                                            file_dir_inp=file_dir_inp)
                dictionary['ID'].append(ID)
                dictionary['DF'].append(corr)
        
        # add ID list
        dictionary['ID'] = [int(x) for x in ids]
            
        return dictionary

#----------------Functional-Data END-------------------------------------------
#----------------Wrap Up-------------------------------------------------------

# Function for prepared data, without rescaling for ML pipeline
def get_behav_connec(conn_name = 'SBC_01', variable_of_interest = 'IAT_sum', name_of_interest = 'IAT'):
    '''
    
    Function to get finalized data to work with

    Parameters
    ----------
    conn_name : str, optional
        Name of the analysis in the CONN-Toolbox. The default is 'SBC_01'.
    variables_of_interest : str, optional
        The behavioral measures which should be left in the final dataframe. 
        The default is ['IAT_sum'].
    names_of_interest : str, optional
        The new names for the behavioral measures 
        The default is ['IAT_sum'].    

    Returns
    -------
    data : DataFrame
        Containing both functional and behavioral data, merged by general IDs.

    '''
    
    if os.path.exists(os.path.join(os.getcwd(), 'data/'+conn_name+'_'+name_of_interest+'_ml_data.xlsx')):
        data = pd.read_excel(os.path.join(os.getcwd(), 'data/'+conn_name+'_'+name_of_interest+'_ml_data.xlsx'))
    
    else:
    # Get DFs
        behav = open_behav_data(variable_of_interest = variable_of_interest, name_of_interest=name_of_interest)
        behav = behav[['ID',name_of_interest]]
        socio = sociodem()
        connec_df = wrap_up_connectivity(output = "df", conn_name=conn_name)
        
        # Merge DFs
        data = pd.merge(pd.merge(behav, socio, on="ID", how='outer'), connec_df, on = 'ID', how='outer')
        
        # Prepare DF
        data = data.astype({'ID':'int'})
        data = data.set_index(data['ID'])
        data = data.drop(['ID'], axis = 1)
        
        # Drop NAN
        data = data.dropna()
        
        # Save as .csv
        data.to_excel(os.path.join(os.getcwd(), 'data/'+conn_name+'_'+name_of_interest+'_ml_data.xlsx'))
        
    return data

# Function for data which is prepared to run in ML pipeline
def get_prepared_data(conn_name = 'SBC_01',
                      variable_of_interest = 'IAT_sum',
                      name_of_interest = 'IAT'):
    '''
    
    Parameters
    ----------
    conn_name : str, optional
        Name of the analysis in the CONN-Toolbox. The default is 'SBC_01'.
    variables_of_interest : str, optional
        The behavioral measures which should be left in the final dataframe. 
        The default is ['IAT_sum'].
    names_of_interest : str, optional
        The new names for the behavioral measures 
        The default is ['IAT_sum'].    

    Returns
    -------
    data : DataFrame
        Containing both functional and behavioral data, merged by general IDs.

    '''
    
    # Check if data has already been generated
    if os.path.exists(os.path.join(os.getcwd(), 'data/'+conn_name+'_'+name_of_interest+'_ml_data_prep.xlsx')):
        data = pd.read_excel(os.path.join(os.getcwd(), 'data/'+conn_name+'_'+name_of_interest+'_ml_data_prep.xlsx'))
        
    else:
        # Get Data and preprocess it for analysis
        data = get_behav_connec(conn_name=conn_name, 
                                variable_of_interest = variable_of_interest,
                                name_of_interest = name_of_interest)
    
        # Recode gender (M = 0; F = 1)
        data['gender'].replace({'M': 0, 'F': 1}, inplace=True)
        
        # Recode age bins in cont variable (20-25 = 1, 26-30 = 2, etc.)
        sorted_data = data.sort_values(by='age', ascending=True)
        unique_bins = sorted_data['age'].unique()
        bin_to_num = {bin: i + 1 for i, bin in enumerate(unique_bins)}
        
        # Map the age bins to numeric values
        sorted_data['age'] = sorted_data['age'].map(bin_to_num)
        data = sorted_data.sort_values(by='ID', ascending=True)
        data.to_excel(os.path.join(os.getcwd(), 'data/'+conn_name+'_'+name_of_interest+'_ml_data_prep.xlsx'), sheet_name='data')
    
    return data

#---------------Wrap Up END----------------------------------------------------
