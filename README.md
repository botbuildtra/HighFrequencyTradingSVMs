# High Frequency Trading using Support Vector Machines
This project implements a high frequency trading strategy that utilizes Support Vector Machines to capture statistical arbitrage in the pricing of Class A and Class C Google stocks. 

This idea was inspired by the following paper: http://cs229.stanford.edu/proj2015/028_report.pdf by Jiayu Wu. 

## Abstract
Pairs Trading is a market-neutral trading strategy that matches a long position with a short position in a pair of highly correlated instruments such as two stocks. 
The idea is to wait for a alteration of the spread of the two stocks and then long the underperforming instrument while simultaneously shorting the overperforming instrument. 
The position should be closed once the instruments return to statistical norms, earning the trader a profit. A good pair should share as many the same intrinsic characteristics as possible. 

A canonical example of pairs trading would be with Pepsi and Coca Cola. Barring any drastic shifts in the characteristics of either company, the returns of Pepsi and Coca Cola should be very correlated. 
![alt text](imgs_readme/pairs_graph1.jpg)
