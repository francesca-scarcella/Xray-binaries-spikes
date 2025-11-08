import os
from glob import glob

from pathlib import Path

import h5py
import copy

import numpy as np
import pandas as pd

from NbodyIMRI import tools, particles, simulator, reader
from NbodyIMRI import units as u

##---------------------------##


#fuction to read a single file
def read_hdf5_contents(h5file, target_datasets):
    contents = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name in target_datasets:
            contents[name] = obj[()]
    h5file.visititems(visitor)
    return contents

#read the profile files
def read_profiles(f, d=1):
    '''
    d: read one in d profiles. If d=1 (default) read all files
    '''    
    profile_dict = {}   
    profile_list = []   

    group = f['data']

    for dataset_name in group:
        norbits = dataset_name.split('_')[2]        
        Norbits= float(norbits)

        if Norbits%d==0:
            profile_list.append((Norbits, group[dataset_name][:]))

    sortedlist=sorted(profile_list)

    for Norbits, profile in sortedlist:       
        profile_dict[str(Norbits)] = profile 

    return profile_dict  



 
               
def get_IDs(directory, expected_count=16):
    """Simple version using string replacements."""
    ids = set()
    
    for f in os.listdir(directory):
        if f.endswith('.hdf5'):
            # Remove extensions and modifiers
            id_name = f.replace('.hdf5', '').replace('_profiles', '')
            # Remove all reload_ prefixes
            while id_name.startswith('re_'):
                id_name = id_name[3:]
            ids.add(id_name)
            
    #print(ids)
    
    if expected_count and len(ids) != expected_count:
        print(f"Warning: Expected {expected_count}, got {len(ids)}")
    
    return sorted(ids)


def read_files_single_it(directory, IDhash, data_dict, target_datasets, n_it):
           
        prefix = "re_" * (n_it)
       
        
        # Construct filenames
        main_filename = f"{prefix}{IDhash}.hdf5"
        profiles_filename = f"{prefix}{IDhash}_profiles.hdf5"
    
        
        
        main_path = os.path.join(directory, main_filename)
        profiles_path = os.path.join(directory, profiles_filename)
        
        try:
            with h5py.File(main_path, 'r') as h5file:
                # Store file contents in dictionary with filename as key
                data_dict[IDhash] = read_hdf5_contents(h5file, target_datasets)
            #print(f"Successfully read profiles: {file_name}")
        except Exception as e:
            print(f"Error reading data {main_filename}: {str(e)}")  
            
        try:
            with h5py.File(profiles_path, 'r') as h5file:
                # Read profile dictionary
                profiles = read_profiles(h5file)
                #discard the keys (# orbits) and store the density profiles           
                data_dict[IDhash]["profiles"]= np.asarray(list(profiles.values()))[:,1]
                if n_it==0:
                    data_dict[IDhash]["profile_bins"]=np.asarray(list(profiles.values()))[0,0]
            #print(f"Successfully read profiles: {file_name}")
        except Exception as e:
            print(f"Error reading profiles {profiles_filename}: {str(e)}")
                    
                                       
                
#read files from all simulations
def read_all_files(gamma_list, path_to_folder, target_datasets, dictionary, n_central_bin, r=1):
    '''
    r: number of times the simulation was restarted (default=1)
    ''' 
            
    for gamma in gamma_list:
    
        print(f"reading files for gamma={gamma}")
        
        path= path_to_folder+"results_gamma_" + "%.2f" % gamma    
        directory= os.path.expanduser(path)
        
        ID_list=get_IDs(directory)
               
        data_gamma={}
               
        for IDhash in ID_list:
            
            
            #read first run of simulation
            read_files_single_it(directory, IDhash, data_gamma, target_datasets, n_it=0)            
            #read subsequent runs 
            for i in range(r):                
                n_it=i+1
                temp_dict={}
                read_files_single_it(directory, IDhash, temp_dict, target_datasets, n_it)           
              
               #merge with the main dictionary
                try:
                    for key in {'data/M_1', 'data/M_2', 'data/vBH1', 'data/vBH2', 'data/xBH1', 'data/xBH2', "profiles" }:

                            data_gamma[IDhash][key]=np.concatenate([data_gamma[IDhash][key], temp_dict[IDhash][key][1:]])

                    #for the list of times, add the final time of the first run
                    t_end=data_gamma[IDhash]["data/t"][-1]
                    t_reload= t_end + temp_dict[IDhash]["data/t"][1:]
                    data_gamma[IDhash]["data/t"] =np.concatenate([data_gamma[IDhash]["data/t"], t_reload])
                    
                except Exception as e:
                        print(f"Error reading reload {IDhash}: {str(e)}")

                                                            
        #compute additional parameters and add them to the dictionary
        for IDhash in list(data_gamma.keys()):

            data=data_gamma[IDhash]

            M1=data["data/M_2"][0]
            M2=data["data/M_1"][0]


            data["xBH"]=data["data/xBH1"]-data["data/xBH2"]
            data["vBH"]=data["data/vBH1"]-data["data/vBH2"]

            data["a"],data["e"] = tools.calc_orbital_elements(data["xBH"],data["vBH"], M1+M2 )   
            data["Torb"]= tools.calc_Torb(data["a"], M1+M2)
            
            
            data["rho_c"]=data["profiles"][:,n_central_bin]


            data_gamma[IDhash]=data


        #add to main dictionary
        dictionary[gamma] = data_gamma
        
               
        
def calculate_moving_averages(data_dict, columns_to_smooth, window_size=100):
    """
    Calculate moving averages for multiple dictionary entries.
    
    Args:
        data_dict: Your dictionary containing arrays
        columns_to_smooth: List of keys to calculate moving averages for
        window_size: Rolling window size 
    
    Returns:
        Dictionary with added moving average columns
    """
    # Create DataFrame with all specified columns
    df = pd.DataFrame({col: data_dict[col] for col in columns_to_smooth})
    
    # Calculate moving averages
    for col in columns_to_smooth:
        data_dict[f"({col})_average"] = df[col].rolling(window=window_size).mean()
    
    return data_dict


def calculate_differentials(data_dict, columns_to_diff, timestep):
    """
    Calculate moving averages for multiple dictionary entries.
    
    Args:
        data_dict: Your dictionary containing arrays
        columns_to_smooth: List of keys to calculate diff for 
    
    Returns:
        Dictionary with added diff columns
    """
    
    # Calculate moving averages
    for col in columns_to_diff:
        data_dict[f"diff_[{col}]"]= np.diff(data_dict[col])/timestep
    
    return data_dict


def average_across_realisations(data, keys_to_average):
    """Quick averaging assuming all arrays have same length or can be truncated."""
    data["average"] = {}
    
    sim_ids = [key for key in data.keys() if key != "average"]
    
    for key in keys_to_average:
        # Get all arrays for this key
        arrays = [np.array(data[IDhash][key]) for IDhash in sim_ids if key in data[IDhash]]
        
        if arrays:
            # Find minimum length and truncate
            min_len = min(len(arr) for arr in arrays)
            truncated = [arr[:min_len] for arr in arrays]
            
            # Calculate average
            data["average"][key] = np.mean(truncated, axis=0)
    
    return data
