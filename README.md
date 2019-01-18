# High Frequency Trading using Support Vector Machines
This project implements a high frequency trading strategy that utilizes Support Vector Machines to capture statistical arbitrage in the pricing of Class A and Class C Google stocks. 

We will demonstrate a trading algorithm that earns great profits (~53% returns over 2 years) with a sharpe ratio of 17.

This idea is heavily inspired by the following paper: http://cs229.stanford.edu/proj2015/028_report.pdf by Jiayu Wu. This project includes modifications into an actual trading strategy that uses rebalancing. 

TODO: Further Model Tuning

TODO: Model Explanations

TODO: Results Graphs

TODO: Diagnostic Graphs on labelling

TODO: Labelled profit vs. Unlabelled profit

## Abstract
D.S. Ehrman defines Pairs Trading as a nondirectional, relative-value investment strategy that seeks to
identify two companies with similar trading characteristics whose equity securities are currently trading at a range outside their historical range. 
This investment strategy entails buying the undervalued security while short-selling the overvalued security; thereby maintaining market neutrality.
The position should be closed once the instruments return to statistical norms, earning the trader a profit. A good pair should share as many the same intrinsic characteristics as possible. 

In our project, we will be searching for pairs trading opportunities between Class A and Class C Google stocks. Since all of the underlying fundamentals of both instruments are similar with the exception of voting rights, this pair makes a very good candidate to explore. 

However, since this pair of of instruments is obviously closely related, many other players in the market are ready to profit off of any mispricings within this pair. It is not expected for any mispricings to be available for a long time. As such, we need to work in a timeframe that is as fast-paced as possible. That is why we will be using a fast-paced high frequency pairs trading strategy to capture statistical arbitrage within the pricing of GOOGL and GOOG as soon as they occur. In our project, we will be creating features from the ticker data that we feed into a machine learning model to predict profitable pairs trading opportunities. 

## Dataset
Our dataset contains snapshots of GOOG and GOOGL over the span of roughly 2 years (10/2016 - 11/2018) at the minute-level resolution. 
Our data was gathered from QuantQuote.com, a reputable dealer of fine-resolution ticker datasets. 
Our dataset had some missing values for both tickers. From QuantQuote's website: "Missing data for certain minutes generally means that no trades occurred during that minute." 
We handled this by removing entries from both datasets in which at least 1 of the tickers had a missing entry. The reasoning behind this was that is that pairs trading is impossible in such instances. This only occured for about 0.1% of our dataset. 

![goog_googl](imgs/goog_googl.png)

## Pairs Trading Model
The canonical pairs trading spread model looks like: 

![spreadmodel](imgs/pairs_model.gif)

where ![datat](imgs/datat.gif) represents the returns of instrument ![A](imgs/A.gif) at time ![t](imgs/t.gif) and 
![dbtbt](imgs/dbtbt.gif) represents the returns of instrument ![B](imgs/B.gif) at time ![t](imgs/t.gif). 

![xt](imgs/xt.gif) represents the spread of the returns at time ![t](imgs/t.gif). One of the assumptions of this model 
is the fact that this residual term is mean-reverting. We can assume this especially since the intrinsic characteristics 
of both securities in this instance are very similar. 

![drift](imgs/drift.gif) represents the drift term. Sometimes the spread begins to trend instead of reverting to the original mean. The drift term is one of the biggest factors of risk in pairs trading. For our problem, we assume that the drift term is negligible compared to the returns of either instrument. 

![beta](imgs/beta.gif) represents the hedge ratio which serves to normalize the volatility between the instruments. 
![beta](imgs/beta.gif) tells us how much of instrument ![B](imgs/B.gif) to long/short for every 1 unit of ![A](imgs/A.gif) to long/short, creating a risk neutral position. We will use the close prices to calculate percent returns and for the other features. Past work has considered assumed that ![beta](imgs/beta.gif) remains constant over the duration of the dataset. For our dataset, however, different behavior in the spread is apparent in 2017 and 2018. This might be due to some change of intrinsic characteristics of the instruments. For our solution, we will assume treat ![beta](imgs/beta.gif) as variable and recalculate it periodically. 

## Ornstein-Uhlenbeck (OU) Stochastic Process

The Ornstein-Uhlenbeck Stochastic Process is used in finance to model the volatility of the underlying asset price process. The process can be considered to be a modification of the random walk (Weiner Process) in which the properties of the process have been changed so that there is a tendency to walk back towards a central location. The tendency to move back towards a central location is greater when the process is further away from the mean. Thus, this process is called "mean-reverting", and has many direct applications in pairs trading. 

The OU process satisfies the following stochastic differential equation: 

![ou](imgs/ou.gif)

where ![theta](imgs/theta.gif) > 0, ![mu](imgs/mu.gif), ![sigma](imgs/sigma.gif) represent parameters, 
and ![weiner](imgs/weiner.gif) denotes the Weiner process (standard Brownian motion process).

![Xt](imgs/xt.gif) is the spread of the two instruments at time ![t](imgs/t.gif). 

![theta](imgs/theta.gif) measures the speed of ![Xt](imgs/xt.gif) returning to its mean level, denoted by ![mu](imgs/mu.gif). 

![sigma](imgs/sigma.gif) represents the volatility of the spread. 

In this project, we will start from the difference of returns at time ![t](imgs/t.gif). Then we will 
integrate this process and use a linear regression to estimate the parameters ![theta](imgs/theta.gif),![mu](imgs/mu.gif), and ![sigma](imgs/sigma.gif). 

These parameters are used later for feature generation. 

## Features

Typically, trading strategies will only apply the spread model to the price of the instruments. In this project, however, 
we will extend it to also include the spread of some technical indicators, namely Simple Moving Average (SMA), Exponentially Weighted Moving Average (EWMA), Money Flow Index (Money Flow Index), and Relative Strength Index (Relative Strength Index). 

The calculation for each of these features is detailed here: 

![sma](imgs/sma.png)

![ewma](imgs/ewma.png)

![mfi](imgs/mfi.png)

![rsi](imgs/rsi.png)

We can extend the spread model to include these technical indicators in addition to price because they exhibit similar 
behaviors for both instruments of the pair. Furthermore, it will provide more context to the classifier in addition to the price. 

## Feature Generation Steps

These are the steps to follow to process the data for the Support Vector Machine algorithm to be used.

1. Split dataset into train and test partitions.

2. Calculate features (price, sma, ewma, mfi, rsi) for each instrument.

3. Calculate percent change for each feature.

4. Run linear regression on the features of the training partition of instrument A and B. 
This serves to estimate ![beta](imgs/beta.gif) according to the pairs trading spread model mentioned above. 
It also calculates the residual term ![dxt](imgs/dxt.gif).

5. Construct the spread terms ![xt](imgs/xt.gif) with the following equation: ![dxtxt](imgs/dxtxt.gif).

6. Run lag 1 auto-regression on spread terms to get parameters according to OU model. 

7. Calculate T-scores for each individual feature according to the following equation:

![tscore](imgs/tscore.gif)

8. Transform the testing dataset by finding the spread residuals ![dxt](imgs/dxt.gif) 
by using the ![beta](imgs/beta.gif) obtained in the training set. 

9. Construct the spread terms ![xt](imgs/xt.gif) of the testing dataset using the following equation: ![dxtxt](imgs/dxtxt.gif).

10. Calculate the T-scores for each individual feature in the testing partition by using the ![mu](imgs/mu.gif) 
and ![sigma](imgs/sigma.gif) calculated for the training partition for each feature. 

At the end of this step, we have fitted the pairs trading spread model parameters and the OU process parameters to the training partition. 

Furthermore, we have transformed both the training and testing partitions according to the parameters and calculated the T-score for them. 
