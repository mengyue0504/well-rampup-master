# Shenzi Well Rampup Project
The project helps automate and manage the Standing Instructions PDF file, update the PI-Af attributes. The forecast model recommends the CV setpoint and the estimated time to the next setpoint change.

## Environment
- Python 3.12
- numpy (1.26.0)
- pandas (2.1.4)
- pymupdf (1.23.10)
- sqlalchemy (1.4.39)
- pyodbc (5.0.1)
- scikit-learn (1.5.0)
- matplotlib (3.8.2)
- requests (2.31.0)
- requests-kerberos (0.14.0)
- requests-ntlm (1.2.0)

## Configuration (configuration.py)
Sql connection settings:
    server, database, username, password

## Usage
- Sharepoint Access.bat search and downloads the most recent standing instructions file into "\data\standing_instructions"
- PDF parser.bat runs the standing instructions parsing
- WellRampModel.bat 




