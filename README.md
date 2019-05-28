# PM2.5 Neural Network Regressors


Run classification model by calling:
``` 
python main.py -normalize
```

To run wavelet, use
```
python main.py -normalize --mode 1
```

To run clustering, use
```
python main.py -normalize --mode 2
```

To use hybrid model, use
```
python main.py -normalize -hybrid
```

(Note that hybrid model can be stacked with wavelet or clustering models)
