# Apple Leaf Disease Classification API

## Overview
This repository contains code for an API that classifies apple leaf diseases using computer vision. It utilizes a pre-trained deep learning model to classify images of apple leaves into different disease categories.

## Requirements
To run the Apple Leaf Disease Classification API, you'll need the following dependencies installed:

- gunicorn
- uvicorn
- fastapi
- numpy
- opencv-python
- python-multipart
- starlette
- tensorflow
- typing

You can install these dependencies using the provided `requirements.txt` file.

## Installation
To install the required dependencies, you can use the provided `install.sh` script or manually install them using `pip`:

```bash
pip install -r requirements.txt
```

## Running the API
You can run the API using Gunicorn with the following command:

```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker api_runner:app
```

This command will start the API on `http://127.0.0.1:8000`.

## Usage
This API provides two main functionalities:

- A web-based user interface (UI) for uploading images and viewing classification results.
- An API endpoint for programmatic classification of images.

## API Endpoints
- `/`: The root endpoint serves the web-based UI for uploading images and viewing classification results.
- `/classify`: This endpoint accepts POST requests with image uploads and returns classification results in JSON format.

## Web-Based User Interface (UI)
You can test the API using a web-based user interface (UI) by visiting the following URL in your web browser:

**Test UI URL:** `http://127.0.0.1:8000`

The UI allows you to upload images and view classification results. Below are screenshots of the UI:
<p align="center">
<img src = "screenshots/screenshot1.PNG">
<img src = "screenshots/screenshot1.PNG">
</p>

## Contributing
Contributions to this project are welcome! Please follow the standard GitHub fork and pull request workflow.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
