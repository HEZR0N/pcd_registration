## Usage
1. Download data
```
python3 data_prep.py
```
 - To view pointclouds: `python3 2view_interact.py -h`
2. Run the ICP algo on the data
```
python3 run_ICP.py
```
3. Train the model on the data
```
python3 run_feature_match.py
```
4. Test the model on the data
```
python3 test_feature_match.py
```
