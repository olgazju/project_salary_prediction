# Salary Prediction for Data Professionals

## Overview

In the dynamic world of data science and technology, I find it really essential to understand what drives salaries in this field. This isn't just important for professionals like me but also for those hiring in the sector and those analyzing industry trends. My project is centered around building a machine learning model that can predict the salaries for these diverse roles. I'm diving into complex questions, like how a person's job title, their experience, where they're based, and their company's size, can influence their earnings. Through this project, my aim is really to help people understand how much they can earn in different tech roles like data science, engineering, analysis, and machine learning. It's all about making things clearer for everyone - whether you're trying to figure out your next career move, or you're a company looking to hire the right talent. I want to give a down-to-earth, real-world picture of what pay looks like in these fields, breaking it down so it's easy to get the whole picture of the tech job market's salary trends.

### Objectives

- To analyze how factors like job title, experience level, geographical location, and company size influence salaries in data field.
- To develop a predictive model that accurately estimates the salaries of data professionals based on the aforementioned factors.
- To provide insights into current salary trends in the data science job market.

### Challenges

- Handling a diverse range of factors that influence salary.
- Ensuring accurate and reliable predictions in a rapidly changing job market.
- Dealing with potential data inconsistencies and biases.

## Dataset

### Data Source

The dataset used for this project is sourced from [Kaggle's Data Science Job Salaries dataset](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries).

### Description

The dataset presents a range of factors that potentially influence the salaries of data professionals, spanning the years 2020 to 2023. It includes information such as job titles, job categories, salary in different currencies, standardized salary in USD, employee residence, experience level, employment type, work setting, company location, and company size.

### Features

- `work_year`: The year when the salary was paid.
- `job_title`: Title of the job.
- `job_category`: Category of the job.
- `salary_currency`: The currency of the reported salary.
- `salary`: The amount of salary paid in `salary_currency`.
- `salary_in_usd`: The amount of salary standardized in USD.
- `employee_residence`: Country of residence of the employee.
- `experience_level`: Experience level of the employee (e.g., Entry, Mid, Senior).
- `employment_type`: Type of employment (e.g., Full-time, Part-time).
- `work_setting`: The setting of work (e.g., Remote, In-person).
- `company_location`: The location of the company.
- `company_size`: The size of the company (e.g., Small, Medium, Large).

| work_year | job_title               | job_category                  | salary_currency | salary | salary_in_usd | employee_residence | experience_level | employment_type | work_setting | company_location | company_size |
|-----------|-------------------------|-------------------------------|-----------------|--------|---------------|-------------------|------------------|-----------------|--------------|------------------|--------------|
| 2023      | Data DevOps Engineer    | Data Engineering               | EUR             | 88000  | 95012         | Germany           | Mid-level        | Full-time       | Hybrid       | Germany          | L            |
| 2023      | Data Architect           | Data Architecture and Modeling | USD             | 186000 | 186000        | United States     | Senior           | Full-time       | In-person    | United States    | M            |
| 2023      | Data Architect           | Data Architecture and Modeling | USD             | 81800  | 81800         | United States     | Senior           | Full-time       | In-person    | United States    | M            |
| 2023      | Data Scientist           | Data Science and Research      | USD             | 212000 | 212000        | United States     | Senior           | Full-time       | In-person    | United States    | M            |
| 2023      | Data Scientist           | Data Science and Research      | USD             | 93300  | 93300         | United States     | Senior           | Full-time       | In-person    | United States    | M            |

### Data Preprocessing

Data preprocessing steps include cleaning, handling missing values, encoding categorical variables, and normalizing numerical features. The dataset is then split into training and testing sets for model development and evaluation.

## Instructions on How to Run the Project

### Setting up the Environment

1. First, ensure you have [pyenv](https://github.com/pyenv/pyenv) installed on your system.

2. Install Python 3.11.4 using pyenv:

    ```bash
    pyenv install 3.11.4
    ```

3. Navigate to the project directory:

4. Create a new virtual environment for this project:

    ```bash
    pyenv virtualenv 3.11.4 salary-prediction
    ```

5. Set the local Python version to use the virtual environment you just created:

    ```bash
    pyenv local salary-prediction
    ```

6. Now, the project directory is set up with a local virtual environment using Python 3.11.4.

### Installing Dependencies

1. Install the dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### Setting up Visual Studio Code (VSCode)

1. Open the project directory in VSCode.
2. Ensure you have the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) installed in VSCode.
3. You can now create and open Jupyter Notebooks within VSCode.
4. When you run Jupyter Notebook in VSCode choose salary-prediction Python environment from drop-down list of avaliavle environments.

### Setting up Jupyter Notebook Independently

1. If you prefer to use Jupyter Notebook outside of VSCode, ensure you have Jupyter installed:

    ```bash
    pip install jupyter
    ```

2. Launch Jupyter Notebook from the project directory:

    ```bash
    jupyter notebook
    ```

3. Jupyter Notebook will open in your web browser, and you can create new notebooks or open existing notebooks from the browser interface.

## Exploratory Data Analysis (EDA)

The exploratory data analysis is detailed in **notebook.ipynb**, using the dataset located at `data/jobs_in_data.csv`.

## Model Selection Process and Parameter Tuning

In **notebook.ipynb**, I explore a variety of models: LinearRegression, RandomForestRegressor, SVR, CatBoostRegressor and a simple neural network model.

I performed cross-validation and hyperparamets tuning.

The final fight was between CatBoostRegressor and SVR:

```
CatBoostRegressor:
Mean Absolute Error (Train): 37964.68
Mean Absolute Error Percentage (Train): 0.29
--------------------------------------
R-squared (Validation): 0.37
Mean Squared Error (Test): 2282144154.58
Mean Absolute Error (Test): 36687.85
Mean Absolute Error Percentage (Test): 0.29

SVR
Mean Absolute Error (Train): 37488.94
Mean Absolute Error Percentage (Train): 0.29
--------------------------------------
R-squared (Validation): 0.37
Mean Squared Error (Test): 2309968633.19
Mean Absolute Error (Test): 36906.22
Mean Absolute Error Percentage (Test): 0.30
```

## Training the Final Model

I copied the code from data cleaning and features extraction process into `train.py` file.

It loads the dataset from `data/jobs_in_data.csv`, cleans it and extract festurs. Then it finetune and train the model.

How to run:

```python
python train.py
```

<img width="642" alt="image" src="https://github.com/olgazju/project_salary_prediction/assets/14594349/59c20be7-4225-45cf-81db-d848560fc58b">

## Loading the Model and Serving It via a Web Service

Finally, I've set up a web service (Fast API) to serve the trained model.

The code for the service is located in the `predict.py` file.

### How to Run the Service

To run the web service, use the following command in the terminal:

```bash
uvicorn predict:app --reload
```

Then open `predict.ipynb`. Here you can find code for request to the server (sort of a client).

First it reads a random row from test set and shows it as JSON:
<img width="1002" alt="image" src="https://github.com/olgazju/project_salary_prediction/assets/14594349/31d217ce-5cd0-4a15-af8c-6feda5966e39">

The next cell uses this JSON to make a request to the server:
<img width="566" alt="image" src="https://github.com/olgazju/project_salary_prediction/assets/14594349/f8702244-c898-4723-a906-1f6355bc722c">

Meanwhile, the server side:
<img width="1183" alt="image" src="https://github.com/olgazju/project_salary_prediction/assets/14594349/08b8af3e-d985-4c68-a376-4228b43b61f6">

## Docker

To containerize and run the model locally using Docker, you'll need to follow these steps:

### 1. **Install Docker Desktop for Mac**

- Go to the [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop) download page.
- Click on "Download for Mac" and the download will start automatically.
- Once downloaded, double click on the Docker.dmg file to open it.
- Drag the Docker icon to your Applications folder to complete the installation.
- Launch Docker Desktop from your Applications folder.

### 2. **Build the Docker Image**

- Open a terminal.
- Navigate to the project root directory.
- Run the following command to build `salary-predictor` Docker image:

```bash
docker build -t salary-predictor .
```

<img width="1179" alt="image" src="https://github.com/olgazju/project_salary_prediction/assets/14594349/34fb0424-e7df-4646-abf4-770f041285a9">


### 3. **Run the Docker Image**

- After successfully building the image, run the following command to start a container from the image

```bash
docker run -p 8000:8000 salary-predictor
```
<img width="1187" alt="image" src="https://github.com/olgazju/project_salary_prediction/assets/14594349/a0055861-00c7-45b9-b5d1-49fedd9db58a">


### 4. **Accessing the Service**

- Now that the model is running in a Docker container, open `predict.ipynb`. Here you can find code for request to the server (sort of a client). Follow the instructions from the previous step

<img width="965" alt="image" src="https://github.com/olgazju/project_salary_prediction/assets/14594349/4ace4db4-7816-4049-a837-2d51ee5c000d">


### 5. **Stop the Docker Container**

- To stop the Docker container, find the container ID with the following command:

```bash
docker ps
```

Then stop the container with:

```bash
docker stop <container-id>
```

## Cloud Deployment

