# rec_engine_stream
## usage

poetry install 

poetry run python3 -m rec_engine.run_read_data

## Google local review 
These datasets contain reviews about businesses from Google Local (Google Maps). 
Data includes geographic information for each business as well as reviews. Download the full datasets from 
https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local. 

Users are indexed by `gPlusUserId` and items(place) are by `gPlusPlaceId`. User and item features are contained in 
`users.clean.json` and `places.clean.json`.  For example, for each `gPlusUserId`, we know his/her `jobs`, `education`, 
and `currentPlace`. 

To treat the data as a streaming data, we let the review time `reviewTime` be the time variable `t`. 

## Papers using the data
Two papers are saved in the `paper` folder. 


## Useful links
https://github.com/blei-lab/context-selection-embedding





