# SE-CBIR

This repository aims for a quick check of the results of my Master thesis: "Saliency-Enhanced Content-Based Image Retrieval in Dermatology Imaging. 

To reproduce the results, please follow the preparation steps below: 

1. Clone this repository: 
    > * git clone git@github.com:mgassnerstudent/SE-CBIR.git
    > * cd SE-CBIR

2. install dependencies using the requirements.txt file:
    > * pip install virtualenv
    > * virutalenv se-cbir
    > * source se-cbir/bin/activate
    > * pip install -r requirements.txt

3. Extract data from [polybox](https://polybox.ethz.ch/index.php/s/013sG9EuMJXhUwr "Polybox link")
    * 3ch.h5 and 4ch.h5 into models/
    * HAM/*, saliency_maps/*, and 4ch_input/* into data/
    * 3ch_emb_test_all.json and 4ch_emb_test_all.json to data/embeddings

4. Check SE-CBIR results as described in two chapters
    * Quantitative Evaluation
    * Qualitative Evaluation

## Quantiative Evaluation

## Qualitative Evaluation

