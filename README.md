# HumanClassifier_DecisionTrees
One more Human classifier but with Decision Tree method
*Created by Yaroslav Oliinyk*

		
### Launching the program
1. Download the repository
2. Unzip the repository in "*Downloads*" folder
3. You'll have the following package: HhmanClassifier_DecisionTrees-master
4. Enter this package
5. Unzip the program
6. In the "requirements.txt" files look what libraries you ne to download in order to launch the program
7. Then use: "*python human_classification.py*"
8. Well done! The program is running.

### Interface and controlling
* Firstly, you'll have a name of the author and name of the algorithm
>![Author and algorithm](https://raw.githubusercontent.com/yaroslavoliinyk/HumanClassifier_DecisionTrees/master/pics/1.JPG)
* Then the table will appear. On that table there are comparisons of model precisions: **test_score**, **training_score**, **cross_val_score**
It shows scores after fitting of 3 different sets of data: with the same data we fit, with one test data and with cross validation method  
>![Three methods comparison](https://raw.githubusercontent.com/yaroslavoliinyk/HumanClassifier_DecisionTrees/master/pics/2.JPG)
* You can also see what **depth** of our Decision Tree we chose and the precision.
* After all the calculation done, you'll be able to see a graph, which represents these 3 scores. We chose the best validation score, because it's save from 
overfitting
* **After that the program ends with the resulting pecentage of correct answers**
### Have fun!

		All rights reserved
			2019
