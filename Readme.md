<h1>Ornamentation Suggester [Trills]</h1>

This project uses a CNN autoencoder model to generate trill suggestions from a midi input primer. 
You can upload and convert your own MIDI data in _main.py_ and `model()` will generate a prediction of where a trill 
could go in the midi data. 

The goal of this project is to be able to provide ornamentation suggestions for composers, when they are composing 
music on MIDI supported score notation software. This is currently a **_proof of concept_** **unfinished** project - at the moment, trill
suggestions are given by training a model using the velocities given in MIDI data as a positional marker - which is 
an arduous and painstaking process as you have to create your own training dataset marking the position of trills in a 
scorewriter, e.g. Musescore, Sibelius, or a DAW, e.g. Logic, Ableton. 

**As it stands, the output suggests trills, but also completely changes your MIDI input, **SO TAKE NOTE****.  

## <h>Installation</h>

- Download the repository 'CC/' using 

- cd to 'CC/' and activate virtual environment

- Install required libraries/packages using `pip install -r requirements.txt` in shell
 
 
- Check to make sure the following package is installed: https://github.com/craffel/pretty-midi


### <h>Input</h>
###### <h>MIDI Primer</h>
For generation of trills from a primer MIDI input, you will need to provide a MIDI data file, that can be inputted in  _main.py_. 
A collection of random MIDI files can be found in the _CC/midi_primer_ directory, which can fufill this purpose.

###### <h>Training Dataset</h>

If you would like to train the model using your own dataset of midi files, these files need to be pre-pre-processed,
as in, all the velocities in the MIDI data need to be set to  80 (comes up as 0 in Musescore) for the `X` input, 
and 80 for 'non-trills' and 127 for trills in the `Y` label input. 

These MIDI files should then be uploaded to _'data/'_ and running _preprocess_data.py_ will create a _pickle_ file  
containing all the training data to upload into the model.





### <h>Generation of MIDI using Pre-trained model</h>
If you are using the pre-trained model to generate trill suggestions, upload a MIDI file to _main.py_, and you should
be able to generate a MIDI output containing the trill suggestions marked by velcity values of 127 (as opposed to 'non-trills' - which have a velocity value of 80) which can be shown 
when uploaded to a scorewriter like Musescore.


### <h></h>



### <h>Training</h>

The model is best trained using GPUs with CUDA. 

To train the model on your own dataset, please pre-process your data in the _preprocess_data.py_ script. 
Training takes place in _model.py_ and various hyperparameters can be adjusted. 

Output of the model can be obtained by running the _main.py_ script. This is only possible if there is a 
trained model available called `my_trained_model.pt` in the CC directory. 

