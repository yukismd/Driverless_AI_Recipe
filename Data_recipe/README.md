## Driverless AI - Data Recipe

[Document](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/custom-recipes-data-recipes.html)  
[Data Recipe Template](https://github.com/h2oai/driverlessai-recipes/blob/rel-1.9.1/data/data_template.py)
****
#### textblobを用いたSentiment Scoreの算出
- [sentiment_score_jp.py](./sentiment_score_jp.py)
#### janomeを用いたテキストカラムのトークン化
- [tokenization_japanese.py](./tokenization_japanese.py)
- ユーザー定義辞書の利用（[辞書の例](../test/tokenize_dict/dict_test.csv)） [tokenization_japanese_dict.py](./tokenization_japanese_dict.py)
- ストップワードの指定[tokenization_japanese_stopwords.py](./tokenization_japanese_stopwords.py)
