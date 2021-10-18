# Steps to run the code

## Step 0:
Download the project folder 'Team_Hal9000_MiniP1.zip' and extract the files to a folder called Team_Hal9000_MiniP1

## Step 1:
Create a virtual environment called 'venv' inside the project folder

``` python -m venv venv ```

## Step 2:
Activate virtual environment on your machine

``` venv\Scripts\activate.bat ```

## Step 3:
run the following commands:
``` 
 pip install matplotlib
 pip install pandas
 pip install sklearn 
 ```

## Step 4: 
if not present, add the datasets to the 'datasets' folder. Follow the folder structure below

    .
    ├── Datasets                   
        ├── BBC                    # folder containing BBC datasets from Greene, D. and Cunningham, P. (2006). for task 1
        ├── drug200.csv            # csv file containing datasets for task 2
    ├── Task1
    ├── Task2
    ├── readme.md

## Step 5:
to run task 1:

``` python ./Task1/main.py ```

Results are shown in ./Task1/Deliverables

to run task 2:

``` python ./Task 2/main.py ```

Results are shown in ./Task2/Deliverables. Running this script may take a few minutes due to the iterations.

