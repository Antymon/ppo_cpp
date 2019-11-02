# paste `echo *.csv` > merged.csv
# only newline as separator (not spaces)
ls scalars* | xargs -d '\n' paste > merged.csv