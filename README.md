# Natural language processing course 2022/23: `Literacy situation models knowledge base creation`

Team members:
 * `Nace Gorenc`, `63160013`, `ng0632@student.uni-lj.si`
 * `Jernej Vrhunc`, `63150316`, `jv4739@student.uni-lj.si`
 * `Žan Pečovnik`, `63160406`, `zp8358@student.uni-lj.si`
 
Group public acronym/name: `skupina123`

# Link to shared data in oneDrive
https://unilj-my.sharepoint.com/personal/slavkozitnik_fri1_uni-lj_si/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fslavkozitnik%5Ffri1%5Funi%2Dlj%5Fsi%2FDocuments%2FFRI%2FNLP%5FCourse%5F2023%2FProjects%20data%20to%20share%2FSkupina123

# Instructions for usage  
Instructions for installation:
`pip install allennlp allennlp-models stanza spacy numpy pandas afinn nltk scikit-learn Levenshtein networkx`

Instructions for visualizing:
- open `test_visualization.py`
- change file name to whatever you want to visualize e.g. `The_Cat_Maiden_afinn`
- run in terminal `python test_visualization.py`

Instructions for running sentiment analysis:
- for Stanza
    - uncomment code which has a comment `# stanza` on the right side of code
    - run in terminal `python sentiment_analysis.py`

- for Afinn
    - uncomment code which has a comment `# afinn` on the right side of code
    - run in terminal `python sentiment_analysis.py`

Instructions for computing metrics:
- run `model_analysis.ipynb` notebook
