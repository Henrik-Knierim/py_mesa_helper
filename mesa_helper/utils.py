import os
from typing import Any, Callable
import numpy as np

def validate_file(filename):
    """Check if a file exists and is valid."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")
    
def validate_option(option: Any, options: list) -> None:
    """Checks if option is in options and if not raises an error."""
    if option not in options:
        raise ValueError(f"Option {option} is not in the list of valid options.")

def create_movie(
    png_header: str,
    png_src: str = "png",
    destination: str = "movies",
    movie_name: str = "movie.mp4",
) -> None:
    """Creates a movie from the png files in `png_src` with the header `png_header` and saves it to `destination` with the name `movie_name`."""
    images_from = os.path.join(png_src, png_header) + "*.png"
    images_to = os.path.join(destination, movie_name)

    command = f"images_to_movie '{images_from}' {images_to}"

    os.system(command)

def clean(
    remove_photos: bool = False, photos_to_save : list = [], remove_pngs : bool = False, remove_logs: bool = False, logs_path: str | None = None
) -> None:
    """Removes the photos, pngs and logs from the run."""
    if remove_photos:
        if photos_to_save:
            files = os.listdir("photos")
            for file in files:
                if file not in photos_to_save:
                    os.remove(os.path.join("photos", file))
        else:
            os.system("rm -r photos")

    if remove_pngs:
        os.system("rm -r png")

    if remove_logs:
        if logs_path is None:
            raise ValueError("logs_path is not defined.")
        os.system(f"rm -r {logs_path}")

def single_data_mask(data: np.ndarray, mask_function: Callable | None = None) -> np.ndarray:
    """Returns the data mask for the quantity x."""

    if mask_function is None:
        return np.ones_like(data, dtype=bool)

    return mask_function(data)

def multiple_data_mask(input_data: list[np.ndarray], mask_filters: list[Callable | None] | None = None) -> np.ndarray:
    """Returns the data mask for the input_data."""

    mask = np.ones_like(input_data[0], dtype=bool)
    if mask_filters is None:
        return mask

    for key, filter in zip(input_data, mask_filters):
        mask &= (mask if filter is None else filter(key))
    return mask

def sort_list_by_variable(input_list: list, variable: str | None) -> list:
    """Sorts a list of strings by a variable in the string. If variable is None, returns the original list."""
    
    if variable is None:
        return input_list
    
    def extract_value(item):
        parts = item.split('_')
        try:
            if variable == "first":
                for part in parts:
                    try:
                        value = float(part)
                        break
                    except ValueError:
                        pass

            else:
                index = parts.index(variable)
                value = parts[index + 1]
        
            # Überprüfen, ob der Wert numerisch ist
            float_value = float(value)
            return float_value
        
        except ValueError:
            raise ValueError(f"Der Wert der Variablen '{variable}' ist kein numerischer Wert.")
        except IndexError:
            raise ValueError(f"Die Variable '{variable}' wurde in '{item}' nicht gefunden.")
    
    # Sortieren der Liste nach dem extrahierten Wert
    sorted_list = sorted(input_list, key=extract_value)
    return sorted_list