import os
from typing import Any

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