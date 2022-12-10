# Github page for XRay Fingerprints DeepLearning Model 
## Training a model to determine XRay Source 

### Software Requirements: ###
Specific Specs for the environment are in the environment_stats folder. Run using Python 3.9.15 with miniconda and an Apple M2 Chip 

Steps to reproduce: 
## Preprocessing 
1. Start with preprocessing folder containing 3 pkl files and the Jupyter Notebook preprocessing.ipynb. Run preprocessing until you 
get to STOP HERE. You should have a csv file generated within your directory containing only the relevant data: Xray Id, 
manufacturer, clinical trial site, etc.
2. **Alter csv via $nano metadata_100subset_df.csv and put "id" on top of first column**  

## Copying Data
3. Edit xraynames.py to represent your path/to/metadata_subset.csv and run the python script
4. ** Remove the id header and make text file instead of csv** should now have xraynames.txt in your folder containing all the names of the images you want to download from Discovery
5. transfer this xraynames.txt file as well as the bash script copy_100subset.bash to Discovery in a new folder to keep your xray subsets (/dartfs-hpc/rc/home/t/f006cht/xray_subsets) 
6. run the copy_100subset.bash script within your xray_subsets folder to copy the file names to the current folder 
7. run scp -r your_username@discovery.dartmouth.edu:/dartfs-hpc/your/path/to/xray_subsets . on your local computer in a directory you want your xray files to be placed **this is for local run** 

## Running the Model 
8. alter training_pytorch.ipynb to represent your path names METADATA_SUBSET_PATH and self.img_dir, the metadata csv file containing the labels for each xray image as well as the image folder containing the corresponding images 
9. run training_pytorch.ipynb 






