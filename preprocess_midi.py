import pretty_midi as pm
import pandas as pd
import pickle
import numpy as np
import os


def encode_midi(inputFile):
    # Load midi data
    midi_data = pm.PrettyMIDI(inputFile)

    # Create lists for storing data needed from midi to create dataset
    list_velocities = []
    list_start = []
    list_end = []
    list_pitch = []

    # Get data from midi
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Velocities represent ornaments (used for y labels)
            list_velocities.append(note.velocity)
            # Get data for x input
            list_start.append(note.start)
            list_end.append(note.end)
            list_pitch.append(note.pitch)

    # Trying to get rid of any reasons why nan is appearing:
    # changing data so that there are no 0.0

    # new_start = []
    #
    # # Loop through list of velocities
    # for i in list_start:
    # # If 0.00 is present append to new value
    #     if i == 0.00:
    #         new_start.append(0.001)
    #     else:
    #         new_start.append(i)

    # Store data for midi file in a Pandas dataframe
    data = pd.DataFrame([list_start, list_end, list_pitch, list_velocities])
    data = data.transpose()
    data.columns = ['Start', 'End', 'Pitch', 'Velocities']
    data.head()

    return data


def extract_segments_from_one_dataframe(df):
    # Convert to numpy array
    array = df.to_numpy()
    # Get array height as number
    array_height = array.shape[0]

    # Define segment height
    seg_height = 50

    # Define list for that specific dataframe
    # This will contain all the segments and labels in tuples (see below)
    segments_and_labels_list = []

    # Iterate over numpy array
    for i in range(0, array_height, seg_height):
        # Don't include last few notes if they don't nicely align to 50
        # Only happens once at the end of the loop
        if i + seg_height > array_height:
            break

        # Current segment: From i to i + segment height
        # Will be 50x4
        label_segment = array[i:i + seg_height, :]
        # Option to check if there is a 127.00 velocity in the data
        # for element in label_segment:
        # if element[-1] == 127.0:
        # print("asdf")
        input_segment = np.copy(label_segment)

        # Set trills back to 80 in input
        input_segment[:, 3] = 80

        # Convert the current segment and has_trill into a tuple
        # This will explicitly store the segment and label as a pair and make data access later super easy
        # See https://www.w3schools.com/python/python_tuples.asp
        my_tuple = (input_segment, label_segment)

        # Append tuple of current segment and label to list
        segments_and_labels_list.append(my_tuple)

        # Just a note: Later you can access the tuple elements by
        seg = my_tuple[0]
        label = my_tuple[1]

    return segments_and_labels_list


if __name__ == '__main__':

    # Define the directory containing midi files to create dataset from
    dataDir = '/Users/eleanorrow/Desktop/CC/data'

    # Define list to pickle
    pickle_list = []

    # Loop to go through all midi entries in directory
    for entry in os.scandir(dataDir):
        if entry.path[-4:] == '.mid' or entry.path[-4:] == '.midi':
            dataframe = encode_midi(entry.path)
            print(dataframe)

            segments_and_labels = extract_segments_from_one_dataframe(dataframe)

            # Append all the tuples you just generated to the mama list
            # Extend vs. append bc append will add the "segments_and_labels" as a whole list to "mama_list"
            # So you'd get a list of lists which is ugly
            pickle_list.extend(segments_and_labels)

            # Mama list now contains all the (segment, label) tuples
            # Dump into pickle
            with open('data.pkl', "wb") as output_file:
                pickle.dump(pickle_list, output_file)

    # ......
    # ......
    # ......

    # Load later with
    # with open('data.pkl', "rb") as input_file:
    # pickle_list = pickle.load(input_file)
