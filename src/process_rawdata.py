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

style = {"description_width": "initial"}
d65 = colour.CCS_ILLUMINANTS["cie_10_1964"]["D65"]


####### FUNCTIONS #######


def MFT_Fotonowy(files:list, info_analysis:Optional[bool] = False, save:Optional[bool] = False, return_data: Optional[bool] = False):
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

    Returns
    -------
    _type_
        For each file in the list of files, it returns a tuple of two Pandas dataframes, where the first dataframe contains the reflectance spectra and the second dataframe contains the colorimetric data.
    """

    raw_files_sp = [file for file in files if 'spect_convert.txt' in file]
    
    for raw_file_sp in raw_files_sp:

        raw_file_dE = raw_file_sp.replace('-spect_convert.txt', '.txt') 

        raw_file_path = Path(raw_file_dE)
        folder = raw_file_path.parent
        filename = raw_file_path.stem
                   

        raw_df_sp = pd.read_csv(raw_file_sp, sep='\t', skiprows = 1)
        raw_df_dE = pd.read_csv(raw_file_dE, sep='\t', skiprows = 8)

        # final wavelength range
        wanted_wl = pd.Index(np.arange(410,751), name='wavelength_nm')

        # interpolated spectral counts values
        df_counts_interpolated = pd.DataFrame(index=wanted_wl)

        for col in raw_df_sp.columns:
            counts_interpolated = sip.interp1d(x=raw_df_sp.index, y=raw_df_sp[col])(wanted_wl)
            df_counts_interpolated[col] = counts_interpolated

        # select white and dark spectral references (first and second columns respectively)
        white_ref = df_counts_interpolated.iloc[:,0].values
        dark_ref = df_counts_interpolated.iloc[:,1].values

        # remove the white and dark ref
        cols = df_counts_interpolated.columns
        df_counts = df_counts_interpolated.iloc[:,2:-1]
        df_counts.columns = cols[3:]
        df_counts.index.name = 'wavelength_nm'

        # define parameters for colorimetric calculations
        D65 = colour.SDS_ILLUMINANTS["D65"]
        d65 = colour.CCS_ILLUMINANTS["cie_2_1931"]["D65"]
        cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1931 2 Degree Standard Observer"]

        # calculate the initial Lab values
        counts_i = df_counts.iloc[:,0]
        sp_i = counts_i.values/(white_ref)
        sd_i = colour.SpectralDistribution(sp_i, wanted_wl)
        XYZ_i = colour.sd_to_XYZ(sd_i, cmfs, illuminant=D65)
        Lab_i = np.round(colour.XYZ_to_Lab(XYZ_i / 100, d65),2)

        # final reflectance values
        df_sp = pd.DataFrame(index=wanted_wl)
        df_sp.index.name = 'wavelength_nm'

        # empty list to store XYZ values
        XYZ = []

        # drop the before last column of df_counts
        df_counts = df_counts.drop(df_counts.iloc[:,-2].name,axis=1)

        # compute the reflectance, XYZ, and LabCh values
        for col in df_counts.columns:  
            counts = df_counts[col].values
            sp = counts / white_ref
            sd = colour.SpectralDistribution(sp, wanted_wl)
            df_sp[col[15:]] = sp
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

        
        # Compute the delta E values
        dE76 = np.round(np.array([colour.delta_E(Lab[0], d, method="CIE 1976") for d in Lab]),3)
        dE00 = np.round(np.array([colour.delta_E(Lab[0], d) for d in Lab]),3)

        # Retrieve the times and energy values
        times = [float(col[15:-3]) for col in df_counts.columns]
        He = raw_df_dE['Watts']       # in MJ/m²
        Hv = raw_df_dE['Lux'] * 1000  # in klxh


        # interpolate colorimetric data
        wanted_He = np.arange(0,He.values[-1], 0.1)

        times = np.round(sip.interp1d(x=He, y= times)(wanted_He),2)
        Hv = np.round(sip.interp1d(x=He, y= Hv)(wanted_He),2)
        L = np.round(sip.interp1d(x=He, y= L)(wanted_He),2)
        a = np.round(sip.interp1d(x=He, y= a)(wanted_He),2)
        b = np.round(sip.interp1d(x=He, y= b)(wanted_He),2)
        C = np.round(sip.interp1d(x=He, y= C)(wanted_He),2)
        h = np.round(sip.interp1d(x=He, y= h)(wanted_He),2)
        dE76 = np.round(sip.interp1d(x=He, y= dE76)(wanted_He),2)
        dE00 = np.round(sip.interp1d(x=He, y= dE00)(wanted_He),2)


        # Final colorimetric values
        df_dE = pd.DataFrame({'time_s':times,'He_MJ/m2':wanted_He,'Hv_klxh':Hv, 'L*':L, 'a*':a, 'b*':b, 'C*':C, 'h':h, 'dE76':dE76, 'dE00':dE00})



        df_sp.columns = He
        df_sp = pd.DataFrame(data = np.round(sip.interp2d(df_sp.columns,pd.to_numeric(df_sp.index), df_sp)(wanted_He, wanted_wl),4),
                            index = wanted_wl,
                            columns = wanted_He,
                            )
        df_sp.columns.name = 'He_MJ/m2'

        if info_analysis:
            lookfor = '#Time'
            file_raw_dE = open(raw_file_dE).read()

            parameters = file_raw_dE[:file_raw_dE.index(lookfor)].splitlines()
            dic_parameters = {}

            for i in parameters:             
                key = i[2:i.index(':')]
                value = i[i.index(':')+2:]              
                dic_parameters[key]=[value]
                df_parameters = pd.DataFrame.from_dict(dic_parameters).T            

            df_parameters.loc['[MEASUREMENT DATA]'] = ''
            df_parameters = df_parameters.reset_index()
            df_parameters.columns = [0,1]
            
            df_dE = df_dE.T.reset_index().T
            df_dE.index = np.arange(0,len(df_dE))

            df_dE = pd.concat([df_parameters,df_dE]).set_index(0)
            df_dE.index.name = 'parameter'


        # save the dataframes
        if save:
            df_sp.to_csv(folder / f'{filename}_SP.csv',index=True)
            df_dE.to_csv(folder / f'{filename}_dE.csv',index=False)

        if return_data:
            return df_sp, df_dE
        
        return None



        