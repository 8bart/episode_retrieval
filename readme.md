## Overview

This project includes the files from my master thesis on football analytics that I can make public. During the project, I created an Information Retrieval model that retrieves interesting episodes from Tracking data.
The following image represents a schematic overview of the framework.

![Information Retrieval Model](/images/IR_model.png)

This project makes it possible to retrieve episodes from tracking data. Episodes are situations or moments during a game that typically last between ten and thirty seconds. The workflow is as follows:
1. Features are generated from tracking data (query_features.py)
2. Choose between Temporal Query (Notebooks) or Query by Semantic Example (qbse.py) 
3. Execute Query

Furthermore, there are some notebooks available in the folder notebooks that to retrieve the formation of a team and cluster the episodes. Unfortunately, these notebooks are work in progress.

## Installation

The first step is to clone the repository

```
git clone https://github.com/8bart/episode_retrieval.git
```

It is recommended to use `virtualenv` for development:

- Start by installing `virtualenv` if you don't have it
```
pip install virtualenv
```
- Once installed access the project folder
```
cd episode_retrieval
```
- Create a virtual environment
```
virtualenv venv
```

- Enable the virtual environment
```
source venv/bin/activate
```

- Install the python dependencies on the virtual environment
```
pip install -r requirements.txt
```

- The project makes use of some functions of the LaurieOnTracking repository to handle the tracking data provided by Metrica

```
git clone https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking.git
```


## Project explanation
The project consists of a several folders and files.
 - helpers: Several .py files that function as library for processing tracking data
 - notebooks: Jupyter notebooks that are used for development
 - results: Data files that can be used as input for notebooks or qbse.py file
 - sample-data-master: Tracking data provided by Metrica (source: https://github.com/metrica-sports/sample-data)
 - query_features.py: pipeline scripts that generates features from tracking data
 - qbsy.py: Script that calculates the similarity between two episodes
 
 


