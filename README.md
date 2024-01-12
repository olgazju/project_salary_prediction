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

The exploratory data analysis is detailed in **EDA.ipynb**, using the dataset located at `data/jobs_in_data.csv`.

In the end, the cleaned dataset was saved to **data/cleaned.parquet** for use in Modeling.
