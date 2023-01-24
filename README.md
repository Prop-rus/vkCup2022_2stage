# vkCup2022_2stage
2nd task of Vk_cup 2022/2023 in building recommendations system
This solution took 7th place out of 256 participants and allowed to be one of 16 final contestants

https://vk.com/@vkteam-cup

## Description
It is needed to predict next most intresting articles for subscribers of blogging platform and sort them in order of time they would spend reading.

Train data - the sequence of ids of users and items, that they interacted with. Also data was enriched by time of interaction with each item and reactions of users (like/dislike).
Test data - selection of users to predict their next 20 items.
Meta - for each item the id of author and embedding of its content were given

Metric - ndcg@20

## My solution
For generating candidates I used two recommender models: ALS of implicit library (https://implicit.readthedocs.io/en/latest/als.html) and Harness Universal recommender of Action ML (https://actionml.com/docs/harness_intro).

!Note: there is no code how to install harness server in this repository. If you want to repeat all models of candidates generation, you should install harness server and create engine by your own. Though the json of engine configuration, the script to generate json with events and scrips to request candidates from server are applied here. Also you can run code with only als candidates, after removing 'harn' value from list in settings file.

### ALS
in Als training the sparse matrix of interactions with items consisted of values = 1 + timedacay * timespent. Number of factors = 700, 100 iteractions.

### Harness
Harness events where devided to categories - short read (timespent < 2 min), long read (timespent >= 2 min), like, dislike, non action (timespent ==0 and reaction is null), author id. Event to predict - long read.

### Ranking
for ranking lambdarank of LightGBM library was used. First ranker was trained on validation set, then made predictions for users in test set.

Run script by `python3 main.py`
