# LAMP
Lanugage Automated Machine Learning by Python.<br>
This tool was built for my master's thesis in biotechnology from Harvard University: Extension School. It is a natural language processor (NLP)
that utilizes pytorch, huggingface, and scipy to make predictions of pathogenicity of bacteria 
based off of semantic analysis of scientific abstracts.

## Installation

Use the package manager [pip]

```bash
pip install lamp
```

## Usage

```python
from lamp.sentences import word_stats, word_cloud # trimming algorithms

word_cloud(word_stats(['This is one','This is two']))
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
