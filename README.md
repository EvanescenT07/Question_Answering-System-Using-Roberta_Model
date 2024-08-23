# Question Answering System Using RoBERTa Model

This project is a final submission for a Natural Language Processing (NLP) course. It implements a Question Answering System using the RoBERTa pretrained model and the SQuAD dataset.

## Project Overview

The system leverages the RoBERTa model to answer questions based on the SQuAD dataset. It employs various libraries and tools for effective model training and evaluation, including:

- `wandb` for experiment tracking and visualization
- `simpletransformers` for simplified model interfacing
- `json`, `re`, and `nltk` for data processing and manipulation
- `streamlit` for integrating QA system on the website

## Getting Started

To explore the project and view the output, visit the Google Colab notebook:
[Google Colab Notebook](https://colab.research.google.com/drive/19w78i7DX12QjKwtjjaBC0u9YSPUVddK4?authuser=2#scrollTo=x6Z1MMdbGGtx)

or run the training code locally:

```bash
jupyter notebook QA_Roberta.ipynb
```

## Setup and Dependencies

Ensure you have the following dependencies installed:

- `wandb`
- `simpletransformers`
- `json`
- `nltk`
- `streamlit`
- `torch`

You can install these packages using pip:

```bash
pip -r requirements.txt
```

## Dataset

The project uses the SQuAD dataset in JSON format. Ensure the dataset is available and correctly formatted for the system to function properly.

## Usage

To run the QA system via Streamlit, execute:

```bash
streamlit run app.py
```

This command will start a local server and open the QA system in your default web browser.

## Additional Information

For more details on the implementation, parameters used, and model configuration, refer to the project documentation in the Google Colab notebook.

Feel free to open an issue if you have any questions or feedback!

## License

This project is licensed under the MIT License. See the LICENSE file for details.
