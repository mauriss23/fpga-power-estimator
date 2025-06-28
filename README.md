# fpga-power-estimator
This project presents a novel integration of VLSI design principles with Machine Learning to predict power consumption in FPGA-based digital systems. It provides a fast and accurate method to estimate dynamic and static power based on dataset readings extracted from synthesized hardware designs.
# project overview
With increasing complexity in digital circuits and demand for energy-efficient designs, power estimation becomes crucial at early stages of design. Traditional power analysis is time-consuming and tool-dependent. This project solves that by:

- Generating a dataset of power values from FPGA designs
- Extracting relevant design features (e.g., logic utilization, switching activity)
- Training machine learning models to predict power usage
- Comparing prediction accuracy with actual FPGA power data.
# features
a)Dataset creation from real FPGA design power reports with extracted hardware features (LUTs, FFs, IOs, switching activity, etc.)

b)Extensive experimentation with multiple Regressive machine learning models for as data showed non linear patterns, including:
Random Forest
Decision Tree
Support Vector Regression (SVR)
XGBoost (best performer)

c)Best model: XGBoost Regressor
R² Score (Train): 0.9999989271
R² Score (Test): 0.9417
Mean Squared Error (Test): 0.0332
Mean Absolute Error (Test): 0.0940

d)Model performance measured using standard regression metrics (R², MSE, MAE).
e)Potential use in early-stage FPGA/VLSI design space exploration for power-aware architecture planning.

# | Domain         | Tools/Technologies                       
  | VLSI / FPGA    | Vivado / Quartus / Synopsys (design flow) 
  | Machine Learning | Python, Scikit-learn, Pandas, NumPy
  | Visualization | Matplotlib, Seaborn

# Dataset Description
Each row in the dataset corresponds to a synthesized FPGA design with features like:
- LUT count
- Flip-flop count
- I/O usage
- Operating frequency
- Switching activity
- Power (Total, Dynamic, Static)

# user interface
To make the solution accessible and interactive, a Streamlit web app is also developed, enabling users to input FPGA design parameters and receive real-time power predictions. This tool is valuable for early-stage design space exploration and optimizing power-aware hardware architectures.

# note
This project is a small yet sincere effort to bridge the gap between two powerful domains — VLSI design and Machine Learning. While it's just a beginning, it showcases the potential of integrating hardware-level insights with data-driven approaches for smarter, faster, and more efficient design exploration.

