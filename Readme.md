\## GitHub Repository

https://github.com/Laylawakeup/MITB\_MLE\_assignment1



\## Project Description

This project implements a \*\*data pipeline\*\* following the \*\*Medallion Architecture\*\* (Bronze → Silver → Gold) using \*\*PySpark\*\* inside a \*\*Dockerized environment\*\*.  

The goal is to transform raw loan-related datasets into an \*\*ML-ready feature store\*\* for \*\*loan default prediction\*\*.



\- \*\*Bronze\*\*: Ingest raw CSV files into Parquet with metadata.  

\- \*\*Silver\*\*: Clean, normalize, and cast data into consistent schemas.  

\- \*\*Gold\*\*: Aggregate and join tables into a single customer-level dataset (`customer\_features`).  





