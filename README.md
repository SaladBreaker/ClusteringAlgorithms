# CulsteringAlgorithm

## Naive Implementation:
 Implement the naive greedy approach for hierarchical clustering as discussed in the lecture with time complexity of O(n 3 ) and O(n 2 ) memory usage. You can freely choose the distance function and the linkage method. Do not use a built-in implementation of hierarchical clustering. Keep track of the order in which the clusters are merged and build a hierarchy of clusters. Test your algorithm, e.g., by comparing your results to the results obtained by publicly available implementations. 


## Efficient Implementation 
-Implement one of the more efficient algorithms, either based on priority queues or nearest neighbor chains. Use the same distance and linkage method as in the naive approach. Test your implementation thoroughly. 

Option 1: Implement the hierarchical clustering algorithm utilizing priority queues to achieve O(n 2 log n) time. You may choose between the variant using a single priority queue and the variant using a priority queue for every object. Further implementation details for the first method are described in [1]; refer to Chapter 17 of [2] for details on the second approach. Do not use a built-in implementation of hierarchical clustering. You may implement a binary heap by yourself or use available priority queue data structures such as heapq. 1 Note that both methods require updating and deleting entries in the heap. Finding these entries efficiently often requires maintaining an additional data structure. 1https://docs.python.org/3/library/heapq.html 1 

Option 2: Implement the hierarchical clustering algorithm utilizing a nearest-neighbor chain (pseudocode available at [2]). With the use of the nearest neighbor chain, we can reduce the complexity to O(n 2 ). Do not use a built-in implementation of hierarchical clustering, but collections such as deque2 are allowed. 

## Project Testing
In project development, there is the possibility of problems or scenarios arising unforeseen during the architectural development phase. Testing modalities that you should consider/implement for your project are as follows:
● functional testing modalities of the project (unit testing, integration testing, etc.);
● non-functional testing modalities of the project (scalability, performance,
security, cross-platform portability, etc.);
● possible failures of the systems on which the project depends (internet, database,
etc);
● ways to automate the whole testing process;
● the use of a bug tracking or task tracking system for the project implementation
("issues" on github or bitbucket).

## Experimental Evaluation and Documentation
 •  Evaluate your implementations with a focus on runtime.
 • Use randomly generated data and the credit card dataset of the Kaggle notebook.
 The website also contains detailed information on how to use hierarchical clustering in Python.
 • Compare your implementations to those available in Python libraries.
 • Verify your algorithm by comparing on a small subset of the data and visualize your result. Evaluate the runtimes.
 • Is the observed runtime in agreement with the theoretical analysis? Try to explain your findings. Describe your hardware in terms of cache and memory sizes and try to reason about runtime behavior in a discussion of your results. 


# Instalation

* Step 1: Make sure that a Python 3 environment was set up onto the machine. A free version can be found on the official website - https://www.python.org/. More details on how to set up a python environment can be found at: https://www.tutorialspoint.com/python/python_environment.htm. 
* Step 2: Install the required libraries via terminal. The command used for this is: 	pip install pandas scikit-learn numpy heapq . This will enable the application to be run and compiled by the machine.
* Step 3: This application is designed to run in a Jupyter notebook environment. Please make sure you have Jupyter installed and running. To install Jupyter, run the following command in your terminal or command prompt: pip install jupyter.  Once jupyter is installed u can start it by running the following command in the terminal or command prompt: jupyter notebook. After that u are ready to run the application

