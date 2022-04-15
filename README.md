# SE-CBIR

This repository aims for a quick check of the results of my Master thesis: "Saliency-Enhanced Content-Based Image Retrieval in Dermatology Imaging. 

To reproduce the results, please follow the preparation steps below: 

1. Clone this repository: 
    <code><pre>git clone git@github.com:mgassnerstudent/SE-CBIR.git  <br/>
    cd SE-CBIR  </code></pre>

2. install dependencies using the requirements.txt file:
    <code><pre>pip install virtualenv <br/>
    python3.6 venv -m se-cbir <br/>
    source se-cbir/bin/activate <br/>
    pip install -r requirements.txt </code></pre><br/>

3. Extract data from [polybox](https://polybox.ethz.ch/index.php/s/013sG9EuMJXhUwr "Polybox link") <br/>
This is only needed for in depth analysis, such as recalculating the deep feature space, its distances, and the model performances. Basic reproduction of results can be done without this step. <br />
    * 3ch.h5 and 4ch.h5 into models/
    * HAM/*, saliency_maps/*, and 4ch_input/* into data/
    * 3ch_emb_test_all.json and 4ch_emb_test_all.json to data/embeddings

4. Check SE-CBIR results as described in two chapters. <br/>
The results will be saved in the results folder but are also printed to the terminal. 
    * Qualitative Evaluation
    <code><pre>cd quantitative_evaluation<br/>
    python qualitative_evaluation.py </code></pre>
    * Quantitative Evaluation
    <code><pre>cd quantitative_evaluation<br/>
    python quantitative_evaluation.py </code></pre>
    For reproducing the deep feature space and recalculating the distances using cosine similarity.
    <code><pre>    python quantitative_evaluation.py -e  </code></pre>
    For recalculating the distances using cosine similarity without reproducing teh deep feature space. 
    <code><pre>    python quantitative_evaluation.py -i </code></pre>
    For recalculating the model performances 
    <code><pre>    python quantitative_evaluation.py -m </code></pre>




