####### IMPORT PACKAGES #######

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import scipy.interpolate as sip
import colour



####### DEFINE GENERAL PARAMETERS #######

cmap = "viridis"

ccs_1964 = colour.CCS_ILLUMINANTS['cie_10_1964']['D65']
ill_D65 = colour.SDS_ILLUMINANTS['D65']
cmfs_1964 = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"]

ccs_1931 = colour.CCS_ILLUMINANTS['cie_2_1931']['D50']
ill_D50 = colour.SDS_ILLUMINANTS['D50']
cmfs_1931 = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"]

observer1 = '10deg'
illuminant1 = 'D65'

observer2 = '2deg'
illuminant2 = 'D50'

d65 = colour.CCS_ILLUMINANTS["cie_10_1964"]["D65"]


####### FUNCTIONS #######


def MFT_fotonowy(files:list, info_analysis:Optional[bool] = False, save:Optional[bool] = False, return_data: Optional[bool] = False, interpolation:Optional[str] = 'He', step:Optional[float | int] = 0.1):
    """Process the microfading rawdata obtained with a microfading device from Fotonowy

    Parameters
    ----------
    files : list
        A list of files containing rawdata

    info_analysis : Optional[bool], optional
        Whether to include information about the measurements, by default False

    save : Optional[bool], optional
        Whether to save the data as a txt file, by default False

    return_data : Optional[bool], optional
        Whether to return the data, by default False

    interpolation : Optional[str], optional
        Whether to interpolate the spectral and colorimetric values, by default 'He'
        'He' -> interpolation over the radiant energy values in MJ/m**2
        'Hv' -> interpolation over the exposure dose values in Mlxh
        'time' -> interpolation over the duration of exposure in seconds
        'none' -> no interpolation

    step : Optional[float | int], optional
        Interpolation step in the unit defined by the choice of the interpolation scale, by default 0.1

    Returns
    -------
    _type_
        For each file in the list of files, it returns a tuple of two Pandas dataframes, where the first dataframe contains the reflectance spectra and the second dataframe contains the colorimetric data.
    """

    # define parameters for colorimetric calculations
    D65 = colour.SDS_ILLUMINANTS["D65"]
    d65 = colour.CCS_ILLUMINANTS["cie_10_1964"]["D65"]
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"] 

    # wanted wavelength range
    wanted_wl = pd.Index(np.arange(380,781), name='wavelength_nm')
    
    # retrieve counts spectral files to be processed
    raw_files_counts = [Path(file) for file in files if 'spect_convert.txt' in Path(file).name]
    

    # process each spectral file
    for raw_file_counts in raw_files_counts:
        
        # retrieve the corresponding colorimetric file
        raw_file_cl = Path(str(raw_file_counts).replace('-spect_convert.txt', '.txt'))                         

        # upload raw files into dataframes
        raw_df_counts = pd.read_csv(raw_file_counts, sep='\t', skiprows = 1)
        raw_df_cl = pd.read_csv(raw_file_cl, sep='\t', skiprows = 8)        

        # round up the first and last wavelength values
        raw_df_counts.rename(index={380.024:380},inplace=True)
        raw_df_counts.rename(index={779.910:780},inplace=True)        

        # select white and dark spectral references (first and second columns respectively)
        white_ref = raw_df_counts.iloc[:,0].values
        dark_ref = raw_df_counts.iloc[:,1].values

        # remove the white and dark ref        
        df_counts = raw_df_counts.iloc[:,2:-1]
        df_counts.columns = raw_df_counts.columns[3:]

        # rename the index column
        df_counts.index.name = 'wavelength_nm'               

        # create an empty dataframe for the reflectance values        
        df_sp = pd.DataFrame(index=raw_df_counts.index)
        df_sp.index.name = 'wavelength_nm'        

        # drop the before last column of df_counts
        df_counts = df_counts.drop(df_counts.iloc[:,-2].name,axis=1)
        
        # compute the reflectance values
        for col in df_counts.columns:  
            counts = df_counts[col].values
            sp = counts / white_ref            
            df_sp[col[15:]] = sp            
                
        # retrieve the times and energy values
        #times = [float(col[15:-3]) for col in df_counts.columns]
        times = raw_df_cl['#Time']
        interval_sec = int(np.round(times.values[3] - times.values[2],0))
        numDataPoints = len(times)        
        duration_min = np.round(times.values[-1] /60, 2)
        He = raw_df_cl['Watts']       # in MJ/m²
        Hv = raw_df_cl['Lux']         # in Mlxh
        total_He = np.round(He.values[-1],3)
        total_Hv = np.round(Hv.values[-1],3)
        

        if interpolation == 'none':            
            df_cl = np.round(raw_df_cl,3)
            
        else:
            # define abscissa units
            abs_scales = {'He': He, 'Hv': Hv, 'time': times}
            abs_scales_name = {'He': 'He_MJ/m2', 'Hv': 'Hv_Mlxh', 'time': 't_sec'}

            # define the abscissa range according to the choosen step value
            wanted_x = np.arange(0, abs_scales[interpolation].values[-1], step)

            # create a dataframe for the energy and time on the abscissa axis        
            df_abs = pd.DataFrame({'t_sec':times, 'He_MJ/m2': He,'Hv_Mlxh': Hv})
            df_abs = df_abs.set_index(abs_scales_name[interpolation])

            # create an interp1d function for each column of df_abs
            abs_interp_functions = [sip.interp1d(df_abs.index, df_abs[col], kind='linear', fill_value='extrapolate') for col in df_abs.columns]

            # interpolate all columns of df_abs simultaneously
            interpolated_abs_data = np.vstack([f(wanted_x) for f in abs_interp_functions]).T

            # Create a new DataFrame with the interpolated data
            interpolated_df = pd.DataFrame(interpolated_abs_data, index=wanted_x, columns=df_abs.columns)

            interpolated_df.index.name = abs_scales_name[interpolation]
            interpolated_df = interpolated_df.reset_index()

            # modify the columns names according to the choosen abscissa unit
            df_sp.columns = abs_scales[interpolation]
                

            # interpolate the reflectance values according to the wavelength and the abscissa range
            df_sp = pd.DataFrame(data = np.round(sip.interp2d(df_sp.columns,pd.to_numeric(df_sp.index), df_sp)(wanted_x, wanted_wl),4),
                                index = wanted_wl,
                                columns = wanted_x,
                                )
                
            # name the columns
            df_sp.columns.name = abs_scales_name[interpolation]   

            # empty list to store XYZ values
            XYZ = []

            # calculate the LabCh values
            for col in df_sp.columns:
                sd = colour.SpectralDistribution(df_sp[col], wanted_wl)
                XYZ.append(colour.sd_to_XYZ(sd, cmfs, illuminant=D65))        

            XYZ = np.array(XYZ)

            Lab = np.array([colour.XYZ_to_Lab(d / 100, d65) for d in XYZ])
            LCh = np.array([colour.Lab_to_LCHab(d) for d in Lab])
                    
            L = []
            a = []
            b = []
            C = []
            h = []

            [L.append(np.round(i[0],3)) for i in Lab]
            [a.append(np.round(i[1],3)) for i in Lab]
            [b.append(np.round(i[2],3)) for i in Lab]
            [C.append(np.round(i[1],3)) for i in LCh]
            [h.append(np.round(i[2],3)) for i in LCh]

                
            # compute the delta E values
            dE76 = np.round(np.array([colour.delta_E(Lab[0], d, method="CIE 1976") for d in Lab]),3)
            dE00 = np.round(np.array([colour.delta_E(Lab[0], d) for d in Lab]),3)

            
            # create the colorimetric dataframe
            df_cl = pd.DataFrame({'L*': L,
                                'a*': a,
                                'b*': b,
                                'C*': C,
                                'h': h,
                                'dE76': dE76,
                                'dE00': dE00})
                

            # concatenate the energy values with df_cl
            df_cl = pd.concat([interpolated_df,df_cl], axis=1)

            # round up values in df_cl
            df_cl = np.round(df_cl, 3)

        # add information about the analysis
        if info_analysis:
            lookfor = '#Time'
            file_raw_cl = open(raw_file_cl).read()

            parameters = file_raw_cl[:file_raw_cl.index(lookfor)].splitlines()
            dic_parameters = {}

            for i in parameters:             
                key = i[2:i.index(':')]
                value = i[i.index(':')+2:]              
                dic_parameters[key]=[value]
                df_parameters = pd.DataFrame.from_dict(dic_parameters).T            

            df_parameters.loc['duration_min'] = duration_min
            df_parameters.loc['interval_sec'] = interval_sec
            df_parameters.loc['numDataPoints'] = numDataPoints 
            df_parameters.loc['totalDose_MJ/m2'] = total_He
            df_parameters.loc['totalDose_Mlxh'] = total_Hv
            
            df_parameters.loc['[MEASUREMENT DATA]'] = ''
            df_parameters = df_parameters.reset_index()
            df_parameters.columns = [0,1]            
            
            df_cl = df_cl.T.reset_index().T
            df_cl.index = np.arange(0,len(df_cl))            

            df_cl = pd.concat([df_parameters,df_cl]).set_index(0)
            df_cl.index.name = 'parameter'
        
                    
        # save the dataframes
        if save:
            folder = raw_file_cl.parent
            filename = raw_file_cl.stem

            if interpolation == 'none':
                df_sp.to_csv(folder / f'{filename}_SP.csv',index=True, header=True)
            else:
                df_sp.to_csv(folder / f'{filename}_SP.csv',index=True, header=True, index_label=f'wl_nm-vs-{abs_scales_name[interpolation]}')

            if info_analysis:
                df_cl.to_csv(folder / f'{filename}_CL.csv',index=True, header=True)
            else:
                df_cl.to_csv(folder / f'{filename}_CL.csv',index=False, header=True)

        # return the dataframes
        if return_data:
            return df_sp, df_cl
        
        return None



        