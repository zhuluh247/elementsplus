**EcoVision Analytics: UK Renewable Energy Prototype**
**Student:** Uche Victor Oparaoma (250124435)
1. Product Overview
EcoVision Analytics is a proof-of-concept data science product developed for non-technical stakeholders in the UK renewable energy sector. It enables users to visualise historical energy generation trends and execute machine learning forecasts using an intuitive, code-free interface.

2. Installation and Deployment Guide
To install and deploy this product locally, please follow these exact steps:

- Install Python Environment:
Navigate to python.org/downloads.
Download Python 3.10 (or higher).
Crucial Step: During installation, you must check the box labelled "Add Python to PATH". Failure to do this will prevent command line execution.
Extract the Prototype:
Extract the contents of this .zip file to a dedicated directory on your local machine.

- Install Dependencies:
Open your system's Command Prompt (Windows) or Terminal (macOS).
Navigate to the extracted folder using the cd command (e.g., cd Desktop/CETM46_Oparaoma_250124435_Prototype).
Execute the following command to install the required analytical libraries:pip install -r requirements.txt

- Execute the Application:
In the same terminal window, launch the local server by running:streamlit run app.py
The application will automatically open in your default web browser.

3. System Architecture
Backend/Data Layer: Python (Pandas, NumPy)
Analytics/ML Layer: Scikit-Learn (Ridge Regression)
Presentation Layer: Streamlit
