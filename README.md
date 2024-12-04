# score-plotting
Python package to manage plotting and reporting tools for the UFS-RNR workflow.
score-plotting leverages the score-db database which uses the PostgreSQL database
system hosted by AWS on an RDS instance (currently administered by PSL).

# Installation and Environment Setup
1. Clone the score-hv, score-db, and score-plotting repos.  All of these repos are required.
score-db is responsible for managing the backend database which stores 
diagnostics data related to UFS-RNR experiments.  score-db has several APIs
to help users insert and collect data from the score-db database.  score-hv,
on the other hand, is responsible for harvesting data from the diagnostic.
score-plotting contains scripts for plotting statistics such as gsi stats and file counts.

```sh
$ git clone https://github.com/noaa-psl/score-hv.git
$ git clone https://github.com/noaa-psl/score-db.git
$ git clone https://github.com/noaa-psl/score-plotting.git
```

2. For testing and development, we recommend creating a new python environment 
(e.g., using [mamba](https://mamba.readthedocs.io/en/latest/index.html) as shown below or other options such as conda). To 
install the required dependencies into a new environment using the micromamba 
command-line interface, run the following after installing mamba/micromamba:

```sh
$ micromamba create -f environment.yml; micromamba activate score-plotting-default-env
```

3. Install score-hv using [pip](https://pip.pypa.io/en/stable/). From the score-hv directory, run the following:

```sh
$ pip install . # default installation into active environment
```

4. Configure the PostgreSQL credentials and settings for the score-db by
creating a `.env` file and by inserting the text shown below (note: this
text is taken straight from the file `.env_example`).  You will need to 
acquire the database password from the administrator (Sergey Frolov).
Note: this MUST be done before installing score-db, or your database
credentials will not work.

```
from the score-db repo top level, cat the example file
$ cat .env_example

SCORE_POSTGRESQL_DB_NAME = 'rnr_scores_db'
SCORE_POSTGRESQL_DB_PASSWORD = '[score_db_password]'
SCORE_POSTGRESQL_DB_USERNAME = 'ufsrnr_user'
SCORE_POSTGRESQL_DB_ENDPOINT = 'psl-score-db.cmpwyouptct1.us-east-2.rds.amazonaws.com'
SCORE_POSTGRESQL_DB_PORT = 5432
```

5. Install score-db using [pip](https://pip.pypa.io/en/stable/). From the score-db directory, run the following:

```sh
$ pip install . # default installation into active environment
```

6. Depending on your use case, you can install score-plotting using one of three methods using 
[pip](https://pip.pypa.io/en/stable/),

```sh
$ pip install . # default installation into active environment`
```
```sh
$ pip install -e . # editable installation into active enviroment, useful for development`
```
```sh
$ pip install -t [TARGET_DIR] --upgrade . # target installation into TARGET_DIR, useful for deploying for cylc workflows (see https://cylc.github.io/cylc-doc/stable/html/tutorial/runtime/introduction.html#id3)`
```
