# Information extraction
# Requirements
pip install -r requirements.txt
# Classification
## Training
python class_train.py --embeddings<Path to a npz file with the embedding vectors data/vectors or use build_data.py> --save<save model training>
# Named Entity Recognition Model
## Training
python ner_train.py --embeddings <Path to a npz file with the embedding vectors data/vectors or use build_data.py> --save <save model training>
# Relationship Extraction Model
## Training
re_model/train.py

#Information extraction
 Classification -> Named Entity Recognition -> Relationship Extraction
 Use web framework for building APIs with Python: FastAPI
 Install fastAPI -> run python main.py
  
