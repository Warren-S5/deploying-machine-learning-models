import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from regression_model import __version__ as _version
from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


# float type for np.nan
def get_first_cabin(row: Any) -> Union[str, float]:
    """Extracts the first assigned value for cabin variable"""
    try:
        return row.split()[0]
    except AttributeError:
        return np.nan

def get_title(row: str) -> str:
    """Extracts the title (Mr, Ms, etc) from the name variable"""
    line = row
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'

def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """ Preprocessing steps for data """

    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].apply(get_first_cabin)
    data['title'] = data['name'].apply(get_title)
    data['fare'] = data['fare'].astype('float')
    data['age'] = data['age'].astype('float')

    data.drop(labels=config.model_config.dropped_vars, axis=1, inplace=True)

    return data

def load_raw_data(*, data_path: str) -> pd.DataFrame:
    """Loads data from URL"""
    dataframe = pd.read_csv(Path(data_path))
    transformed = preprocessing(preprocessing)

    return transformed

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """ Loads data in data directory"""
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = preprocessing(preprocessing)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
