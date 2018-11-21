# TwiNotify
Before running the entire project, you may want to make sure that you have the following libraries in place: 
1) pandas
2) numpy
3) re
4) tweepy
5) datetime
6) sklearn
7) scipy

How to install: 
If you are using Jupyter Notebook: 
!pip install (library)
 If you are using some other editor, execute the following on the terminal: 
pip install (library) 

Usage: 
1) pandas:
In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It is free software released under the three-clause BSD license.
2) numpy:
NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
3) re:
Regular expressions (called REs, or regexes, or regex patterns) are essentially a tiny, highly specialized programming language embedded inside Python and made available through the re module. Using this little language, you specify the rules for the set of possible strings that you want to match; this set might contain English sentences, or e-mail addresses, or TeX commands, or anything you like. You can then ask questions such as “Does this string match the pattern?”, or “Is there a match for the pattern anywhere in this string?”. You can also use REs to modify a string or to split it apart in various ways.
4) tweepy:
Tweepy is open-sourced, hosted on GitHub and enables Python to communicate with Twitter platform and use its API. 
5) datetime:
The datetime module supplies classes for manipulating dates and times in both simple and complex ways. While date and time arithmetic is supported, the focus of the implementation is on efficient attribute extraction for output formatting and manipulation. 
6) sklearn: 
Scikit-learn (formerly scikits.learn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
7) scipy:
SciPy is a free and open-source Python library used for scientific computing and technical computing.
SciPy contains modules for optimization, linear algebra, integration, interpolation, special functions, FFT, signal and image processing, ODEsolvers and other tasks common in science and engineering.

EXECUTION: 
1) Create a Developer Twitter account – Get real time Twitter API 
Reference: https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html
2) Create a Twilio Account and purchase a phone number. 
Reference: https://www.twilio.com/sms
4) Download the dataset and the Jupyter Notebook from the github
Reference: https://github.com/kaleketan07/TwiNotify
3) Update these fields in the given code: 
consumer_key = 'XXXXXXXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXX'
4) Download ngrok
Reference: https://ngrok.com/download
5) Enter the links on the Twilio portal
6) Send message using your phone to registered Twilio number and get the latest trending alerts. 

Re-execute the application:
Kill the kernal and restart it. 

