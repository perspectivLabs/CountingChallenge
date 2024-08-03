# CountingChallenge

## Notes
    Install the libraries in requirements
    put the two data folders in data/raw/
    For AI install trex by running these
        cd AI
        cd T-Rex
        pip install dds-cloudapi-sdk==0.1.1
        pip install -v -e .
    

## outputs
    outputs are stored in outputs directory in each AI and Non_AI folders
    counts are in counts.txt file in the respective folder
    Masks for both folders are in appropriate folders

## Solutions
    Refer to the demo.ipynb in both AI and Non_AI folder for the solutions code
    scripts and configs are in src

### AI
    The solution is done using T-Rex2 api
    For more details on it refer to https://github.com/IDEA-Research/T-Rex
    Created 2 image prompts to be used as refernce
    The prompts and their creation are in AI/T-rex2/Scratchpad.ipynb
    These prompts are basically two rectangles drawn in two of the bolts images indicating a bolt/screw

### Non_AI
    Used opencv countoring method 
    Worked well for the 1st folder but badly for the second folder


## Additional Notes :
    Tried gronding_dino, grounded_sam,dinov but they are not able to detect these small objects with text prompts.
    count_gd,tfoc(training free object count) is another image prompt model which is open source and worked well but little less accurate compared to T-rex2.
    For Non_AI there are some research and libraries for blood cell counting which are similar to this task.
    Tried Codi-M (a density based approach) but has some memory complexity issues when no of objects to detect are high.