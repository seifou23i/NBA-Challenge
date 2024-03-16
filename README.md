# NBA Player Longevity Prediction
This repository tackles a machine learning challenge: building a model to predict whether an NBA player, based on their statistics, will last over 5 years in the league and whether they are worth investing in.
## Table of contents
The Readme file is organized as follows: TODO


## Dataset Description
The dataset contains various statistical features of NBA players. Here's a table summarizing the variables:

| Variable Name | Description                                                                                                                                 |
|---|---------------------------------------------------------------------------------------------------------------------------------------------|
| Name | Player's name                                                                                                                               |
| GP | Number of games played in the season                                                                                                        |
| MIN | Minutes played per game (average)                                                                                                           |
| PTS | Points scored per game (average)                                                                                                            |
| FGM | Field successful goal attempts per game (average)                                                                                           |
| FGA | Field goal attempts per game (average)                                                                                                      |
| FG% | Field goal percentage (percentage of successful attempts per game)                                                                          |
| 3P Made | Three-point shots made per game (average)                                                                                                   |
| 3PA | Three-point shots attempted per game (average)                                                                                              |
| 3P% | Three-point percentage (percentage of successful 3P per game)                                                                               |
| FTM | Free throws made per game (average)                                                                                                         |
| FTA | Free throw attempts per game (average)                                                                                                      |
| FT% | Free throw percentage (percentage of successful FT per game)                                                                                |
| OREB | Offensive rebounds per game (average): number of times the offensive player grabs the ball after a missed shot attempt by his team          |
| DREB | Defensive rebounds per game (average): number of times the defending player grabs the ball after a missed shot attempt by the opposing team |
| REB | Total rebounds per game (average): offensive rebounds plus defensive rebounds                                                               |
| AST | Assists per game (average): number of passes made by the player that directly lead to a basket scored by a teammate                         |
| STL | Steals per game (average) : number of times the player successfully takes the ball away from an opponent without fouling                    |
| BLK | Blocks per game (average) : number of times the player deflects a shot attempt made by an opposing player                                   |
| TOV | Turnovers per game (average): number of times the player loses possession of the ball due to a mistake or being stripped by an opponent     |
| TARGET 5Yrs | Binary variable indicating playing for at least 5 years (1) or less (0)                                                                     |

**Note:**

* Field goal percentage (FG%) is calculated as FGM / FGA.
* Three-point percentage (3P%) is calculated as 3P Made / 3PA.
* Free throw percentage (FT%) is calculated as FTM / FTA.
* Total rebounds (REB) is calculated as OREB + DREB.
* Games played (GP) and minutes played (MIN) are restricted to realistic values (GP <= 82, MIN <= 42).

This table provides a quick overview of the data available in [nba_logreg.csv](./data/inputs/nba_logreg.csv).

## Exploratory Data Analysis (EDA)
An Exploratory Data Analysis (EDA) report has been generated to understand the data in more detail. This report provides insights into the distribution of features, missing values, correlations, and other statistical summaries.

You can view the detailed EDA report in the generated HTML file: [nba_player_report.html](./data/outputs/nba_player_report.html) 

