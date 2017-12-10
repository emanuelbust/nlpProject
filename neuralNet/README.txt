To repartition the dataset for testing, training, and development, run python splitData.py ../groupChat.txt
To regenerate word embeddings (necessary for training), run python splitData.py ../groupChat.txt
To train the model, execute train.sh (You may need to give yourself permission to execute).
To test the model , execute train.sh (You may need to give yourself permission to execute).

Training and testing will output two lists to the screen, one that is accuracy by epoch and another that is losses by epoch.

**Results text files are contained in the results directory
