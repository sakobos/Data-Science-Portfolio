## Project Description <br>
This project was designed to predict batted ball exit velocity from 4 key metrics (Tilt, Release Height, Induced Vertical Break, & Horizontal Break) measured by the Trackman ball flight tracking system that is installed at the Coastal Carolina baseball stadium. <br>
This project was completed as my Capstone project for the completion of my undergraduate degree in applied physics and includes a PowerPoint presentation and research paper. <br>

## Data <br>
The data for this project was acquired from the Coastal Carolina Baseball program with permission, from their Trackman system. The dataset consisted of every pitch thrown in competition since the installation of the Trackman system in 2017 until the data was given to me in the fall of 2019. <br>
The data was separated into 4 sets using only Coastal Carolina pitchers. Left-Handed Pitcher 4-seam Fastballs (LHP4S), Left-Handed Pitcher 2-seam Fastballs (LHP2S), Right-Handed Pitcher 4-seam Fastballs (RHP4S), and Right-Handed Pitcher 2-seam Fastballs (RHP2S).

## Model <br>
Multiple-factor Linear Regression was used to predict the batted ball exit speed.

## Results <br>
There was no statistically significant relationship found between the 4 chosen factors (Tilt, Release Height, Induced Vertical Break, & Horizontal Break) and the exit velocity of a batted ball. 

## Analysis <br>
There are many other avenues to explore from the data that might be able to provide better results. The data was originally broken up into 4 groups, but the best use of this data would be to separate the data by the individual pitcher, as each pitcher throws in their own unique way that may make their pitches behave/perform differently than another pitcher of the same handedness. Many other measurements can be taken into account as well, such as pitch speed, location, extension, and many more that could all potentially explain the effectiveness of a pitcher's offering.  

## Disclaimer <br>
I was very, very new to the R programming language, and programming in general. I was also balancing other classes and Division-1 athletics at the time, which only granted me so much time to focus on this project. That being said, the code is quite rough, so I apologize for its readability, but I have made very few changes to the original code, only adding comments to help the reader follow along more easily. <br>
