To run the model, you may open command prompt and type  
`python3 Driver.py <directory of training set> <directory of test set> <algorithm> [<learning rate> <max epoch>]`  
The last two arguments are optional and are default to 0.5 and 10, respectively. For `<algorithm>`, please type either `perceptron`
or `voted_perceptron` or `averaged_perceptron`. Any other input will raise error.  
A `SettingWithCopyWarning` may appear when loading the datasets. However, it does not affect the performance of the
algorithms.