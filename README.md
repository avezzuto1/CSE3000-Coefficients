This codebase has been used to obtain the results presented in the thesis paper of the CSE3000 Research Project course.

The graphs.ipynb file can be used to produce the same plots as shown in the paper. The code blocks read the data from the file's output folder and present it in an aggregated transparency scatter plot. Each code block shows graphs for either the relevance-based or item-based concordance coefficients. Lastly, two code blocks at the end summarise the absolute differences between the base and relevance-based coefficients, for the TREC and simulated data respectively.
New data (without ties) can be uploaded to the untied_data folder, and all coefficients are obtained by running the main.py file. This outputs a CSV containing all coefficients for each pairwise comparison. 
