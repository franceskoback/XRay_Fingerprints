import os
import pandas as pd 
import numpy as np

df_path= '/Users/franceskoback/Documents/XRay_Fingerprints/Preprocessing/metadata_100subset_df.csv'
df=pd.read_csv(df_path)
df["id"] = df["id"].astype("str").str.zfill(8)
df["id"] = "/dartfs-hpc/rc/home/t/f006cht/scratch/OAI/processed_images/xrays/knee/BilatPAFixedFlex/224x224/no_dicom_proc/self_scaled/group_norm/"+df["id"]+".npy"
id_column = df.iloc[:, 0]
#xray_id = os.path.basename(img_path).replace(".npy", "")

#print(id_column.head())

for i in id_column.head():
    print(i)
    
id_column.to_csv(r'xraynames.csv',index=False)





#for i in os.listdir():
#    print(i)