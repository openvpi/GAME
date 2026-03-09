# GAME: Generative Adaptive MIDI Extractor

## Overview

GAME is the upgraded successor of [SOME](https://github.com/openvpi/SOME), designed for transcribing singing voice into music scores.

### Highlights

1. Generative boundary extraction: **trade off quality and speed** through D3PM (Structured Denoising Diffusion Models in Discrete State-Spaces).
2. Adaptive architecture: notes and pitches can **align and adapt to known boundaries**.
3. Robust model: **works on dirty or separated voice** mixed with noise, reverb or even accompaniments.
4. Multilingual support: choose the right language or a similar one to improve the segmentation results.
5. Thresholds of boundaries and note presence are adjustable.
6. Produces floating point pitch values, same as what SOME does.

### Use cases

1. Transcribe unlabeled raw singing voice waveforms into music scores, in MIDI format.
2. Align notes to labeled word boundaries, in dataset processing scenarios.
3. Estimate note pitches from note boundaries adjusted by user in interactive tuning tools.

## Installation

GAME is tested under Python 3.12, PyTorch 2.8.0, CUDA 12.9, Lightning 2.6.1. But it should have good compatibility.

Step 1: You are recommended to start with a clean, separated UV or Conda environment with suitable Python version.

Step 2: Install the latest version of PyTorch following its [official website](https://pytorch.org/get-started/locally/).

Step 3: Run:

```bash
pip install -r requirements.txt
```

Step 4: If you want to use pretrained models, download them from [releases](https://github.com/openvpi/GAME/releases) or [discussions](https://github.com/openvpi/GAME/discussions).

## Inference

### Transcribe raw audio files

The inference script can process single or multiple audio files.

```bash
python infer.py extract [path-or-directory] -m [model-path]
```

By default, MIDI files are saved besides each audio file in the same directory. Text formats (.txt and .csv) are also supported.

For example, transcribing all WAV files in a directory:

```bash
python infer.py extract /path/to/audio/dir/ -m /path/to/model.ckpt --glob *.wav --output-formats mid,txt,csv
```

For detailed descriptions of more functionalities and options, please run the following command:

```bash
python infer.py extract --help
```

### Process singing voice datasets

The inference script is compatible with [DiffSinger dataset format](https://github.com/openvpi/MakeDiffSinger). Each dataset contains a `wavs` folder including all audio files, and a CSV file with the following columns: `name` for item name, `ph_seq` for phoneme names, `ph_dur` for phoneme durations and `ph_num` for word span. The script can process single or multiple datasets.

```bash
python infer.py align [path-or-glob] -m [model-path]
```

For example, processing single dataset:

```bash
python infer.py align transcriptions.csv -m /path/to/model.ckpt --save-path transcriptions-midi.csv
```

Processing all datasets matched by glob pattern:

```bash
python infer.py align *.transcriptions.csv -m /path/to/model.ckpt --save-name transcriptions-midi.csv
```

Prediction results are inserted (or replaced) into the CSV: `note_seq` for note names, `note_dur` for note durations, `note_slur` for slur flags; `note_glude` will be removed from CSV because the model does not support glide types.

For detailed descriptions of more functionalities and options, please run the following command:

```bash
python infer.py align --help
```

> [!IMPORTANT]
>
> #### Notice for v/uv flags and word-note alignment
>
> Word boundaries have slightly different definitions between DiffSinger and GAME:
>
> - In DiffSinger, some special unvoiced tags like `AP` (breathing) and `SP` (space) are considered as independent words, with boundaries between them.
> - In GAME, consecutive unvoiced notes are merged into whole unvoiced regions, with no boundaries inside.
>
> To improve the alignment of v/uv flags between words and notes, we should also merge consecutive unvoiced words before inference. This process is done automatically by the inference API and will not affect the original phoneme sequence. For better comprehension, here is an examples of v/uv flags and word-note alignment:
>
> ```text
> ph_seq       | n  |  i   | h  |      ao       |  SP  |   AP   |  => phoneme names
> ph_dur       |0.05| 0.07 |0.05|     0.16      | 0.07 |  0.09  |  => phoneme durations
> ph_num       | 1  |     2     |       1       |  1   |   1    |  => word spans
> word_dur     |0.05|   0.12    |     0.16      | 0.07 |  0.09  |  => word durations
> word_vuv     | 0  |     1     |       1       |  0   |   0    |  => word v/uv
> word_dur_m   |0.05|   0.12    |     0.16      |     0.16      |  => word durations (after merging)
> word_vuv_m   | 0  |    1      |       1       |      0        |  => word v/uv (after merging)
> note_seq     | C4 |    C4     |  D4   |  E4   |      E4       |  => note names (predicted)
> note_vuv     | 0  |    1      |   1   |   1   |       0       |  => note v/uv (predicted)
> note_dur     |0.05|    0.12   | 0.08  | 0.08  |     0.16      |  => note durations (predicted)
> note_seq_a   |rest|    C4     |  D4   |  E4   | rest |  rest  |  => note names (after alignment)
> note_dur_a   |0.05|    0.12   | 0.08  | 0.08  | 0.07 |  0.09  |  => note durations (after alignment)
> note_slur    | 0  |     0     |   0   |   1   |  0   |   0    |  => note slur flags (after alignment)
> ```
>
> By default, a word is considered as unvoiced if its leading phoneme hits a built-in unvoiced phoneme set, and note v/uv flags are predicted by the model. This logic can be controlled through the following options:
>
> - `--uv-vocab` and `--uv-vocab-path` defines the unvoiced phoneme set.
> - `--uv-word-cond` sets the condition for judging a word as unvoiced.
>   - `lead` (default): If the leading phoneme is unvoiced, the word is unvoiced. This is enough for most cases because normal words start with vowels. In this mode, you only need to define special tags in the unvoiced phoneme set.
>   - `all`: If all phonemes are unvoiced, the word is unvoiced. This is the most precise way to judge unvoiced words, but you need to define all special tags and consonants in the unvoiced phoneme set.
> - `--uv-note-cond` sets the condition for judging a note as unvoiced.
>   - `predict` (default): Note u/uv flags are predicted by the model and decoded with a threshold.
>   - `follow`: Note u/uv flags follow word v/uv flags. If you use this mode, you still need to define all special tags and consonants in the unvoiced phoneme set (because sometimes the first word only has one consonant in it).
> - `--no-wb` bypasses all logic above, with no word-note alignment, and everything is purely predicted by the model. Also, no `note_slur` column will be written since the word information is unavailable. Not recommended.

## Training

### Data preparation

1. Singing voice dataset with labeled music scores. Each subset includes an `index.csv`. File structure:

   ```text
   path/to/datasets/
   ├── dataset1/
   │   ├── index.csv
   │   ├── waveforms/
   │   │   ├── item1-1.wav
   │   │   ├── item1-2.wav
   │   │   ├── ...
   ├── dataset2/
   │   ├── index.csv
   │   ├── waveforms/
   │   │   ├── item2-1.wav
   │   │   ├── item2-2.wav
   │   │   ├── ...
   ├── ...
   ```

   Each `index.csv` contains the following columns:

   - `name`: audio file name (without suffix).
   - `language` (optional): code of the singing language, i.e. `zh`.
   - `notes`: note pitch sequence split by spaces, i.e. `rest E3-3 G3+17 D3-9`. Use `librosa` to get note names like this.
   - `durations`: note durations (in seconds) split by spaces, i.e. `1.570 0.878 0.722 0.70`.

2. Natural noise datasets (optional). Collect any types of noise or accompaniments and put them into a directory. Be careful not to include singing voice or clear speech voice.

3. Reverb datasets (optional). Put a series of Room Impulse Response (RIR) kernels in a directory, usually in WAV format. [MB-RIRs](https://zenodo.org/records/15773093) is recommended.

### Configuration

This repository uses an inheritable configuration system based on YAML format. Each configuration file can derive from others through `bases` key. Also, in preprocessing, training and evaluation scripts, configurations can be overridden with dotlist-style CLI options like `--override key.path=value`. 

Most training hyperparameters and framework options are stored in [configs/base.yaml](configs/base.yaml), while model hyperparameters and data-related options are stored in [configs/midi.yaml](configs/midi.yaml). You can also organize your own inheritance structure.

Configure your dataset paths in the configuration:

```yaml
binarizer:
  data_dir: "data/notes"  # <-- singing voice dataset with labeled music scores

training:
  augmentation:
    natural_noise:
      enabled: true  # <-- false if you don't use natural noise
      noise_path_glob: "data/noise/**/*.wav"  # <-- natural noise datasets
    rir_reverb:
      enabled: true  # <-- false if you don't use reverb
      kernel_path_glob: "data/reverb/**/*.wav"  # <-- reverb datasets
```

The default configuration trains a model with ~50M parameters and consumes ~20GB GPU memory. Before proceeding, it is recommended to read the other part of the configuration files and edit according to your needs and hardware.

### Preprocessing

Run the following command to preprocess the raw dataset:

```bash
python binarize.py --config [config-path]
```

Please note that only singing voice dataset and its labels are processed here. The trainer uses online augmentation, so you need to carry everything inside your singing voice, noise and reverb datasets if you need to train models on another machine.

### Training

Run the following command to start a new training or resume from one:

```bash
python train.py --config [config-path] --exp-name [experiment-name]
```

By default, checkpoints and lightning logs are stored in `experiments/[experiment-name]/`. For other training startup options, run the following command:

```bash
python train.py --help
```

You can start a TensorBoard process to see metrics and validation plots:

```bash
tensorboard --logdir [experiment-dir]
```

After validation, you can reduce the size of checkpoint by dropping optimizer states for inference only. Run the following command:

```bash
python reduce.py [input-ckpt-path] [output-pt-path]
```

### Evaluation

Model evaluation uses the same dataset structure, format and configuration file as training. Be sure to use the same feature arguments as the model to evaluate. It is also recommended to read the evaluation configuration:

```yaml
training:
  validation:
	# ...
```

Run the following command to preprocess the test dataset in evaluation mode:

```bash
python binarize.py --config [config-path] --eval
```

Run the following command to evaluate the model on your dataset:

```bash
python evaluate.py -d [dataset-dir] -m [model-path] -c [config-path] -o [save-dir]
```

You can find a `summary.json` in the output directory containing all metric values. If `--plot` option is given, comparison plots will be saved in `plots` folder. For other evaluation startup options, run the following command:

```bash
python evaluate.py --help
```

## Deployment

Models can be exported to ONNX format for further deployment.

### Export ONNX models

Run the following command to export a model:

```bash
python deploy.py -m [model-path] -o [save-dir]
```

By default, ONNX models are exported using _trace_ exporter and opset version 17 for best compatibility. Sometimes you may need higher opset versions for more operators, i.e. native `Attention` operator starting from opset version 23. For opset version 18 or above, it is recommended to use TorchDynamo exporter. For example:

```bash
python deploy.py -m [model-path] -o [save-dir] --dynamo --opset-version 23
```

However, using TorchDynamo and higher opset versions can break compatibilities with some Execution Providers (like DirectML). Please use with caution and test them after exporting.

### Inference with ONNX models

We don't provide implementation of ONNX model inference pipeline in this repository. However, you can read the [documentation](ONNX.md) about the workflow and structures, which may help you understand and implement it.

## Integration

For secondary development or downstream integration, this repository exposes essential APIs of all its stages. Please read the following code for details:

- Preprocessing: [preprocessing/api.py](preprocessing/api.py)
- Training: [training/api.py](training/api.py)
- Inference and evaluation: [inference/api.py](inference/api.py)
- Deployment: [deployment/api.py](deployment/api.py)

## Disclaimer

Any organization or individual is prohibited from using any functionalities included in this repository to generate someone's singing or speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

## License

GAME is licensed under the [MIT License](LICENSE).
