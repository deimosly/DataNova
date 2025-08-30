from PyInstaller.utils.hooks import copy_metadata

# Include Streamlit's metadata
datas = copy_metadata('streamlit')
