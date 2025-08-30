DataNova – Gaia Stellar Classification Tool
===========================================

Welcome, astronomers! DataNova is a Python-based tool for fetching Gaia star samples, training a RandomForest classifier, filtering stellar data, and exporting results for further analysis. This guide will help you set up and run the application.

Install Anaconda
------------------

1. Download the latest version of Anaconda for your system: 
   https://www.anaconda.com/download/success
2. Follow the installer instructions to complete the installation.

   Note: Anaconda provides a stable Python environment with package management and ensures all dependencies work correctly.

Install Required Python Packages
---------------------------------

1. Launch Anaconda Prompt from your system search bar.
2. Run the following command (replace the path with your install_requirements.py location):

   python "C:\path\to\install_requirements.py"

3. The script will check for necessary packages (pandas, numpy, matplotlib, scikit-learn, streamlit, astroquery) and install any missing ones.
4. Wait for all installations to complete. Press Enter when prompted to close the installer console.

Launch DataNova
-----------------

1. Open a new Anaconda Prompt.
2. Run DataNova using:

   streamlit run "C:\path\to\datanova.py"

3. A browser tab will automatically open with the DataNova interface.

   The interface allows you to fetch Gaia samples, train classifiers, explore HR diagrams, filter stars, and export CSV files for analysis.

Finishing Your Session
------------------------

1. Once finished, simply close the browser tab.
2. Close the console window to end the program.

Note:
-----

DataNova is designed for professional or research-level use. Ensure you have a stable internet connection when fetching Gaia data. Large samples provide more accurate classification results.
