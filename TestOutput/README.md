# PyUltraLight
This folder is the default location for saved outputs from the main code, including energy, density and field configuration outputs. In the accompanying Jupyter notebook, the save location may be changed by altering the following parameter:

save_path="TestOutput"

Each time the code is run, a new folder will be created within this TestOutput directory, named according to the formula: "date_time_resolution". Within that folder all chosen output files will be created, as well as an additional config.txt file, which specifies the configuration used for that particular simulation run. 

