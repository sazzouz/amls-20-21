# UCL ELEC0134 AMLS 20/21

### Changed made after deadline

As shown in the commit history, changes have been made right after the deadline. These changes were simply immediate resolutions to the mistaken addition to the .gitignore file of important sudirectories found in each task folder, namely the 'results/
, 'logs/' and 'saved_models/' directories. This had to be resolved otherwise the program will error when these directories are not found. Due to the unforeseen file size of dependencies such as the 68point landmark detector, these changes were delayed as GitHub limits the rate and upload size. I hope this can be understood. Thanks.
### zceesaz@ucl.ac.uk (secondary email for this GitHub account)
Please email me if any issues occur setting up this project

### SN: 15043735

---
## Project structure

This project is organised for the four tasks. Each subfolder contains a task orchestration file, i.e. 'a1.py' followed by a local 'preparation.py' file to prepare the data for this task.

There is also a 'common' folder found in the root of the project to provide modules used in each task.

The progression through each task is handled in 'main.py' where results are tracked and information is provided in the terminal throughout the program

To being the program, please plase the data in the root 'Datasets' folder as per the inital template.

Then, in the root of the directory, simply runt 'python main.py'

## Dependencies

This project was built using Python 3.8+

All packages used in this project can be installed from the 'requirements.txt' file in the root of the directory using pip. This file was generated using the python manager tool called 'Poetry' https://python-poetry.org/