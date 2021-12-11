# neural networks using gaussian initiation
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 5 100 'gaussian'
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 10 100 'gaussian'
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 25 100 'gaussian'
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 50 100 'gaussian'
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 100 100 'gaussian' 0.01 0.03

# neural networks using zeros initiation
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 5 100 'zeros'
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 10 100 'zeros'
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 25 100 'zeros'
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 50 100 'zeros'
python3 driver.py ./bank-note/train.csv ./bank-note/test.csv 100 100 'zeros'