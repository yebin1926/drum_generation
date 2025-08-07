#!/bin/bash
set -e

# Fix locale issue for papermill / click
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

papermill step_1_midi_data_preprocessing.ipynb step_1_output.ipynb
papermill step_2_generate_drum_ssm_from_melodic_ssm.ipynb step_2_output.ipynb
papermill step_3_extract_bar_selection_info.ipynb step_3_output.ipynb
papermill step_4_generate_drum_pattern.ipynb step_4_output.ipynb
papermill step_5_convert_data_into_MIDI.ipynb step_5_output.ipynb