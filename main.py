import argparse
import ollama
from pathlib import Path
import json
from waggle.plugin import Plugin
import logging
import os
import base64
from urllib.parse import urlparse
import subprocess


def get_image_data(image_uri: str) -> bytes:
    scheme = urlparse(image_uri).scheme
    if scheme in ["http", "https"]:
        return get_image_data_http(image_uri)
    if scheme in ["rtsp"]:
        return get_image_data_rtsp(image_uri)
    return get_image_data_file(image_uri)


def get_image_data_http(image_uri: str) -> bytes:
    from urllib.request import urlopen
    from http.client import HTTPResponse
    from http import HTTPStatus
    logging.info("Fetching image from HTTP %s", image_uri)
    with urlopen(image_uri, timeout=30) as resp:
        resp: HTTPResponse
        if resp.status != HTTPStatus.OK:
            raise FileNotFoundError(f"Unable to fetch image from URL: {image_uri}")
        return resp.read()


def get_image_data_rtsp(image_uri: str) -> bytes:
    logging.info("Fetching image from RTSP stream %s", image_uri)
    # Fetch latest frame from RTSP stream using ffmpeg.
    subprocess.check_output([
        "ffmpeg",
        "-rtsp_transport",
        "tcp",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-i", image_uri,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-update",
        "1",
        "-y",
        "output.jpg",
    ])
    return Path("output.jpg").read_bytes()


def get_image_data_file(image_uri: str) -> bytes:
    logging.info("Fetching image from file %s", image_uri)
    return Path(image_uri).read_bytes()


def run(plugin: Plugin, host: str, model: str, prompt: str, images: list[Path]):
    logging.info("Running: model=%r and prompt=%r", model, prompt)

    client = ollama.Client(host=host)

    logging.info("Ensuring model %r has been pulled.", model)
    client.pull(model)

    for image in images:
        logging.info("Processing image: %s", image)

        raw_image_data = get_image_data(image)
        encoded_image_data = base64.b64encode(raw_image_data).decode()

        # Run model on example.
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image_data],
                },
            ],
        )

        # Build output data.
        output = {
            "created_at": response.created_at,
            "load_duration": response.load_duration / 1e9,
            "prompt_eval_count": response.prompt_eval_count,
            # convert from nanoseconds to seconds
            "prompt_eval_duration": response.prompt_eval_duration / 1e9,
            "eval_count": response.eval_count,
            # convert from nanoseconds to seconds
            "eval_duration": response.eval_duration / 1e9,
            "model": response.model,
            "output": response.message.content,
            "input": str(image),
            "prompt": prompt,
        }

        output_json = json.dumps(output, separators=(",", ":"), sort_keys=True)

        logging.info("Publishing results: %s", output_json)
        plugin.publish("ollama_response", output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="enable debug level logging"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "ollama.default.svc.cluster.local"),
        help="ollama host",
    )
    parser.add_argument("-m", "--model", default="gemma3", help="model to use")
    parser.add_argument(
        "-p", "--prompt", default="Describe this image.", help="prompt to use"
    )
    parser.add_argument("images", nargs="*", help="images to process")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with Plugin() as plugin:
        run(
            plugin=plugin,
            host=args.host,
            model=args.model,
            prompt=args.prompt,
            images=args.images,
        )
