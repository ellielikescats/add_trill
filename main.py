import math
import numpy as np
import pandas as pd
import pretty_midi
import torch
import pretty_midi as pm
from DataLoader import CustomMidiDataset
from model import CNNet

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNet()
model = model.to(device)
state_dict = torch.load('my_trained_model.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
dataset = CustomMidiDataset()


def encode_midi_primer(inputFile):
    """
    Helper function for encoding MIDI file to lists
    :param inputFile: Path to MIDI file
    :return: Pandas dataframe of MIDI information
    """
    midi_primer = pm.PrettyMIDI(inputFile)

    # Create lists for storing data needed from midi to create dataset
    list_velocities = []
    list_start = []
    list_end = []
    list_pitch = []

    # Get data from midi
    for instrument in midi_primer.instruments:
        for note in instrument.notes:
            # Velocities represent ornaments (used for y labels)
            list_velocities.append(note.velocity)
            # Get data for x input
            list_start.append(note.start)
            list_end.append(note.end)
            list_pitch.append(note.pitch)

    normalised_velocities = []
    for i in range(len(list_velocities)):
        normalised_velocities.append(80)

    # Store data for midi file in a Pandas dataframe
    data = pd.DataFrame([list_start, list_end, list_pitch, normalised_velocities])
    data = data.transpose()
    data.columns = ['Start', 'End', 'Pitch', 'Velocities']
    data.head()

    return data


def extract_segments_from_one_dataframe(df):
    """
    Helper function to extract segments from one dataframe
    :param df: Pandas dataframe containing MIDI information
    :return: List of segments
    """
    # Convert dataframe to numpy array
    array = df.to_numpy()
    # Get array height as number
    array_height = array.shape[0]

    # Define segment height
    seg_height = 50

    # Define list for that specific dataframe
    # This will contain all the segments in tuples
    segments_list = []

    # Iterate over numpy array
    for i in range(0, array_height, seg_height):
        # Don't include last few notes if they don't nicely align to 50
        # Only happens once at the end of the loop
        if i + seg_height > array_height:
            break

        # Current segment: From i to i + segment height
        # Will be 50x4
        label_segment = array[i:i + seg_height, :]

        # Append tuple of segment to list
        segments_list.append(label_segment)

    return segments_list


# Load midi primer and convert to tensor
def load_and_convert_to_tensor(inputFile, channel=1):
    """
    Function to convert the MIDI file into a tensor that can be inputted into the trained model.
    :param inputFile: Path to input file
    :param channel: No of channels used in model during training
    :return: Primer input as a tensor
    """

    # Extract midi data
    data = encode_midi_primer(inputFile)

    # Convert to segments
    segments = extract_segments_from_one_dataframe(data)

    # Convert segments to array
    segments = np.array(segments, dtype=np.float32)

    # Normalise
    for column in range(segments.shape[2]):
        maxX = dataset.max_values_X[column]

        segments[:, :, column] /= maxX

    # Convert array to tensor
    tensor = torch.tensor(segments)
    num_segments = tensor.shape[0]

    # Reshape(batch_size, 1, num_rows, num_columns)
    primer = torch.unsqueeze(tensor, 0).reshape(num_segments, channel, 50, 4)

    return primer


def tensor_to_midi(tensor):
    """
    Function to convert the output tensor (created from primer tensor into pre-trained model) into a MIDI file.
    :param tensor: Output tensor from inputting the primer midi-tensor into the model
    :return: MIDI file of output
    """

    # Squeeze tensor
    tensor = torch.squeeze(tensor, dim=1)
    # Convert tensor to numpy array, and detach tensor from device
    if device == 'cuda':
        na = tensor.cpu().detach().numpy()
    else:
        na = tensor.detach().numpy()

    # Un-normalise
    for column in range(na.shape[2]):
        maxX = dataset.max_values_X[column]

        na[:, :, column] *= maxX

    # Convert array back to 2d
    num_segments = na.shape[0]
    num_rows = na.shape[0] * na.shape[1]
    num_cols = na.shape[2]
    na2d = np.zeros((num_rows, num_cols))
    for i in range(num_segments):
        start_idx = i * na.shape[1]
        end_idx = (i + 1) * na.shape[1]
        na2d[start_idx:end_idx, :] = na[i]

    # Truncate pitch decimal points
    for i in range(num_rows):
        na2d[i, 2] = math.trunc(na2d[i, 2])

    # If the velocities contain element > 57, then could assume potential trill
    for element in na2d:
        if element[-1] > 57:
            print("Potential trill in", element)

    for element in na2d:
        if not element[-1] > 57:
            break
        print("Are you sure you want to add trills to this piece of music? Hmm maybe we have different tastes...")


    # Convert the output data to a MIDI file
    piano_output = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for piano instrument
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    # Iterate over note names for each row in the 2-dimensional array
    for row in na2d:
        # Create a Note instance, incorporating the elements from row for each note
        note = pretty_midi.Note(
            velocity=int(row[3]), pitch=int(row[2]), start=row[0], end=row[1])
        # Add note to the piano
        piano.notes.append(note)
    # Add the piano instrument to the PrettyMIDI object
    piano_output.instruments.append(piano)
    # Write out the MIDI data
    piano_output.write('piano_output.mid')


# Get midi input primer (choose from a selection in /primer folder
# (can uncomment "primer" below to see various primer-output results)

primer = load_and_convert_to_tensor('/Users/eleanorrow/Desktop/CC/primer/The Continental.mid')
#primer = load_and_convert_to_tensor('/Users/eleanorrow/Desktop/CC/data/Andante.mid')

# Call to model with primer tensor to give output tensor
output = model(primer)
# Convert output tensor back into midi file to analyse
tensor_to_midi(output)

print('Please check the folder for piano_output.mid, and open in Musescore.')
print('For a better understanding of where to find the highest velocity positions (trill location), it would be easier to open .mid file in a DAW, like Logic, and view the piano roll.')
# Open the midi file into a Musescore

