import argparse
import ollama
import base64
from pathlib import Path
import time
import json
from waggle.plugin import Plugin
import logging
import os
from pydantic import BaseModel


class ImageSummary(BaseModel):
    short_description: str
    detailed_description: str
    objects: list[str]


def run(plugin: Plugin, host: str, model: str, prompt: str, images: list[Path]):
    embed_model = "mxbai-embed-large"

    logging.info("Running: model=%r and prompt=%r", model, prompt)

    client = ollama.Client(host=host)

    logging.info("Ensuring model %r has been pulled.", model)
    client.pull(model)
    client.pull(embed_model)

    for image in images:
        logging.info("Processing image: %s", image)

        # Encode image as Base 64 to pass to Ollama.
        encoded_image = base64.b64encode(image.read_bytes()).decode()

        # Run model on example.
        start_time = time.monotonic()
        response: ollama.ChatResponse = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image],
                },
            ],
            format=ImageSummary.model_json_schema(),
        )
        end_time = time.monotonic()
        chat_duration = end_time - start_time

        image_summary = ImageSummary.model_validate_json(response.message.content)

        start_time = time.monotonic()
        # TODO Publish embedding too. For now, we are just computing to see how long it takes.
        embedding = client.embed(embed_model, image_summary.short_description)
        end_time = time.monotonic()
        embed_duration = end_time - start_time

        # Build output data.
        output = {
            "input": str(image),
            "model": model,
            "prompt": prompt,
            "output": json.loads(image_summary.model_dump_json()),
            "chat_duration": round(chat_duration, 3),
            "embed_duration": round(embed_duration, 3),
        }

        output_json = json.dumps(output, separators=(",", ":"), sort_keys=True)

        logging.info("Publishing results: %s", output_json)
        plugin.publish("structured_inference_log", output_json)


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
    parser.add_argument("images", nargs="*", type=Path, help="images to process")
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
