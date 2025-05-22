# Rectok
Codes and data of Rectok.
### Environment
```
conda create -n rectok python=3.10.0
conda activate rectok
pip install -r requirements.txt
```

### Dataset
- download: [text embeddings](https://huggingface.co/datasets/hizkai/text_embeddings)
```
unzip them and move to dataset/ and run tokenization 
```
alternatively you can directly sft with the existed index files 

### Tokenizaion
```
cd Rectok/shell 

- run 
bash get_instruments_token.sh
bash get_games_token.sh 
bash get_baby_token.sh 
``` 
### SFT
``` 
cd Rectok/shell 

- run 
bash run_instruments.sh
bash run_games.sh 
bash run_baby.sh
``` 

