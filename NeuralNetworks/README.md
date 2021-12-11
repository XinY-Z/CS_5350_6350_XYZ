To run the model, you may simply run the `run.sh` file, or open a command prompt and type  

`python3 Driver.py <directory:training set> <directory:test set> <# nodes> <max iteration> <initiation type> 
[<learning rate> <learning d>]`

For `<initiation type>`, type either `gaussian` or `zeros`.
Any other input will raise an error.

The last 2 arguments are optional. `<learning rate>` is default to 0.1. `<learning d>` is default to 0.1. learning
schedule is defined as `<learning rate> / (1 + T * (<learning rate> / <learning d>)`
