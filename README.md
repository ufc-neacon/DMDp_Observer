# DMDp_Observer
 
🗂️ Project Structure and Script Descriptions

File->tank_EKF_example.py

    Purpose: Implements the Extended Kalman Filter (EKF) for the nonlinear tank system.

    Main Tasks:

        Simulates state estimation with realistic measurement noise.

        Saves estimated data for later comparison.

    Execution Order: Run this script first to generate the EKF reference data.


File->tank_polynomial_example.py

    Purpose: Implements the proposed polynomial-based model and compares its performance against DMDc.

    Main Tasks:

        Performs simulations using both methods.

        Plots the results and error between the polynomial model and DMDc.

        Saves data to allow comparison with EKF results.

    Execution Order: Run after tank_EKF_example.py to ensure EKF data is available for comparison.


File->tank_EKFvsDMDp.py

    Purpose: Performs as presented on the research paper,  final comparative analysis between:

        The Extended Kalman Filter (EKF)

        The proposed polynomial-based method

        The DMDc approach




File->pendulum_EKF_example.py

    Purpose: Implements the Extended Kalman Filter (EKF) for the nonlinear undamped pendulum system.

    Main Tasks:

        Simulates state estimation with realistic measurement noise.

        Saves estimated data for later comparison.

    Execution Order: Run this script first to generate the EKF reference data.


File->pendulum_polinomial_example.py

    Purpose: Implements the proposed polynomial-based model and compares its performance against DMDc.

    Main Tasks:

        Performs simulations using both methods.

        Plots the results and error between the polynomial model and DMDc.

        Saves data to allow comparison with EKF results.

    Execution Order: Run after pendulum_EKF_example.py to ensure EKF data is available for comparison.


File->pendulum_EKFvsDMDp.py

    Purpose: Performs as presented on the research paper,  final comparative analysis between:

        The Extended Kalman Filter (EKF)

        The proposed polynomial-based method

        The DMDc approach
    
    Main Tasks:

        Loads saved data from the previous scripts.

        Plots comparative results and errors across all methods.

    Execution Order: ✅ Run this script last, after both pendulum_EKF_example.py and pendulum_polynomial_example.py.


✅ Recommended Execution Flow




Run EKF simulation

->python tank_EKF_example.py

   Run Polynomial and DMDc simulation:

->python tank_polynomial_example.py

   Run full comparison between EKF, DMDc, and the proposed method: 
   
->pyton tank_EKFvsDMDp.py






📌 Notes

    Same Recommended execution flow valid for pendulum ecample.
    
    All generated data (e.g., state estimates, error metrics) are saved automatically in structured formats for future use or analysis.

    Plots provide visual comparisons between the methods, highlighting estimation errors and system behavior differences.


    
