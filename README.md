# iBex
Help detect ibex from trap cameras using deep learning.

## Before you run
Download this file as a zip and move to a folder you want.

### python
Download python for windows 64-bit version 3.6+
https://www.python.org/downloads/windows/
Or follow this article:
https://phoenixnap.com/kb/how-to-install-python-3-windows
Note that you should click yes on install pip

### pip
pip is a little software that handles the other software.
You should have downloaded in the prevoius step, if not, you can download it by itself.
You may follow this article: https://phoenixnap.com/kb/install-pip-windows
* Open the command prop (Shift and right mouse key, "open powershell window here" in the folder you moved the files to).
* run "pip install -r requirements.txt" 

### Download the Deep Learning Weights
The code is basically only a skeleton where the parameters are located in the weights.
The weights file is too big so it's in the shared Google Drive folder in a folder called "model"
Download it (might take a while..) and then put it in a folder called "model" (the model folder should be in the same folder as main_program)
- there are two files in the model folder
## How to run
in the command prop you opened (in the iBex folder), enter

"python main_program\ibex2csv.py <path_to_camera_traps> <batch_size> <accuracy_threshold min> <accuracy_threshold max>
path_to_camera_traps : path to an image folder
batch_size: currently not supported
accuracy_threshold min: the minimum threshold for putting the ibex in not_sure folder
accuracy_threshold max: the max threshold for putting the ibex in not_sure folder

- Example: \Documents\pictures\cute_ibex 1 0.1 0.9
So in the example if the confidence is in (0.1,0.9), it is defnitly an ibex, if not, put it in not_sure.

### if you want only filtering (much faster)
"python main_program\classification\filter_bex.py <path_to_camera_traps> <batch_size> <accuracy_threshold min> <accuracy_threshold max>

# Contact me if needed: galastra[at]post.bgu.ac.il


##TODO:
- Easier running the python code (argument, help, etc.)
- Faster filtering
