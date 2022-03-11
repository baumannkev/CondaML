To run on your localhost 
Prerequesites: have Python installed to your computer, and set to your environemnt variables: https://www.tutorialspoint.com/How-to-set-python-environment-variable-PYTHONPATH-on-Windows

1. Download the repo to your computer

2. Install conda environment on your computer: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

3. Install prerequisite libraries from requirements.txt 

    3.1. In your terminal go to your project folder and type: 'pip install -r requirements.txt'

4. Create a new conda environment called ml as follows in a terminal command line: 

    4.1 In your terminal go to your project folder and type: 'conda create -n ml python=3.10.0'
    
    4.2 Secondly, we will login to the ml environment: 'conda activate ml'

5. Launching the web app.
   To launch the app, type the following into a terminal command line (i.e. also make sure that the hvacapp.py file is in the current working directory): 
   
   5.1. 'streamlit run hvacapp.py' 


In a few moments you will see the following message in the terminal prompt.

> streamlit run hvacapp.py

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501

Network URL: http://10.0.0.11:8501

P.S: If you run into a problem that says "missing module" error, make sure to 'pip install' said module on your local machine.
   
