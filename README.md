## Style transfer

Transfering style yaknow


### Data

Preprocessed data is here: https://drive.google.com/file/d/0B5Y5rz_RUKRmREtLZ2NleHd2bEk/view?usp=sharing  
Raw data is on the NLP machines if you want to rerun the processing scripts.

sequences/historic and sequences/modern contain the sentences as lists of tokids  
word_vectors/tok_to_id.pkl contains the mapping of tokids to tokens


### Code

Scripts for data preprocessing are in `data_wrangling/`

Code for feeding data into a training pipeline is in `data/`

Msc code (progress bars, constants, etc) goes in `msc/`

Models go in `modeling/`

Training, testing (todo), evaluation (todo) are in `main.py`


### Training a model

**NOTE**: you have to be on Peng's andaconda for this to work (I did some stuff that the cluster's TF version doesn't like): `$ export PATH=/scr/pengqi/anaconda2/bin:$PATH`


Do `python main.py -h` to get help. I'm pretty sure the supported args are
  * `data_dir` is a directory like `jacob:/scr/rpryzant/style-transfer/datasets`
  * `out_dir` is where you want to write your checkpoints, logfiles, etc.
  * `-g X` (OPTIONAL) is to run the pipeline on GPU `X`
  * `-flip` specifies whether to flip gradients or not
  * `-ignore` whether to ignore the discriminator entirely

For example, I've trained a flipped, non-flipped, and discriminator-ignoring models already with
```
python main.py datasets/ notflipped -g 0
python main.py datasets/ flipped -flip -g 1
python main.py datasets/ ignored -ignore -g 2
```
The outputs from these runs live at `jacob:/scr/rpryzant/style-transfer/flipped/` and `jacob:/scr/rpryzant/style-transfer/notflipped` if you want to check them out with tensorboard






### related word rebuttals

-- constrained based approach assumes same semantic space => constrains data to be about same stuff
-- "projecting out" that direction might eliminate useful stuff...we want a "soft projection"
