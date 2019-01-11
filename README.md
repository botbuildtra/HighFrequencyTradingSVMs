# High Frequency Trading using Support Vector Machines
This project implements a high frequency trading strategy that utilizes Support Vector Machines to capture statistical arbitrage in the pricing of Class A and Class C Google stocks. 

This idea is heavily inspired by the following paper: http://cs229.stanford.edu/proj2015/028_report.pdf by Jiayu Wu. This project includes modifications into an actual trading strategy that uses rebalancing. 

TODO: Further Model Tuning

TODO: Model Explanations

TODO: Results Graphs

TODO: Diagnostic Graphs on labelling

TODO: Labelled profit vs. Unlabelled profit

## Abstract
Pairs Trading is a market-neutral trading strategy that matches a long position with a short position in a pair of highly correlated instruments such as two stocks. 
The idea is to wait for a alteration of the spread of the two stocks and then long the underperforming instrument while simultaneously shorting the overperforming instrument. 
The position should be closed once the instruments return to statistical norms, earning the trader a profit. A good pair should share as many the same intrinsic characteristics as possible. 

A canonical example of pairs trading would be with Pepsi and Coca Cola. Barring any drastic shifts in the characteristics of either company, the returns of Pepsi and Coca Cola should be very correlated. 

![pairs](imgs/pairs_graph.jpg)

In our project, we will be searching for pairs trading opportunities between Class A and Class C Google stocks. Since all of the underlying fundamentals of both instruments are similar with the exception of voting rights, this pair makes a very good candidate to explore. 

However, since this pair of of instruments is obviously closely related, many other players in the market are ready to profit off of any egregious mispricings within this pair. It is not expected for any mispricings to be available for a long time. As such, we need to work in a timeframe that is as fast-paced as possible. That is why we will be using a fast-paced high frequency pairs trading strategy to capture statistical arbitrage within the pricing of GOOGL and GOOG as soon as they occur. In our project, we will be creating features from the ticker data that we feed into a machine learning model to predict profitable pairs trading opportunities. 

## Dataset
Our dataset contains snapshots of GOOG and GOOGL over the span of roughly 2 years (10/2016 - 11/2018) at the minute-level resolution. 
Our data was gathered from QuantQuote.com, a reputable dealer of fine-resolution ticker datasets. 
Our dataset had some missing values for both tickers. From QuantQuote's website: "Missing data for certain minutes generally means that no trades occurred during that minute." 
We handled this by removing entries from both datasets in which at least 1 of the tickers had a missing entry. The reasoning behind this was that is that pairs trading is impossible in such instances. This only occured for about 0.1% of our dataset. 

![goog_googl](imgs/goog_googl.png)

## Feature Generation
