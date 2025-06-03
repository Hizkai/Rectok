# Rectok
Codes and data of Rectok.
### Environment
```
conda create -n rectok python=3.10.0
conda activate rectok
pip install -r requirements.txt
```

### Dataset
- Download: [datasets](https://huggingface.co/datasets/hizkai/text_embeddings) 
```
unzip them and move to dataset/ and run tokenization 
```
alternatively you can directly run SFT with the existed index files 

### Tokenizaion
```
cd Rectok/shell 

- run 
bash get_instruments_token.sh
bash get_games_token.sh 
bash get_baby_token.sh 
``` 
### SFT
- Download: [Base model](https://huggingface.co/meta-llama/Llama-3.2-1B) 
``` 
cd Rectok/shell 

- run 
bash run_instruments.sh
bash run_games.sh 
bash run_baby.sh
``` 

