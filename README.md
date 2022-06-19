The program was developed using virtualenv environment and it is suggested (but should not be necessary) to use it
Python version: Python 3.6.9
Necessary packages are inside requirements.txt file
To run the program: 
* execute command ```pip3 install -r requirements.txt```
* run ```python3 TextClassifier.py```

Program has 3 options, to use them write either 1, 2 or 3 and press ENTER
Option 1:
* To train the model unzip data provided with the exercise so mlarr_text is in the same folder as TextClassifier.py
* Inside mlarr_text there are folders: business, entertainment, politics etc. and text samples are inside them

Option 2:
* Model is saved in model_params.txt and test data is in the same structure as in option 1

Option 3:
* Model is saved in model_params.txt
* write path to the text file which will be classified for example: ```mlarr_text/business/b_400.txt```
