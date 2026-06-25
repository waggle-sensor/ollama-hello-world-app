import argparse
import ollama
from pathlib import Path
import json
from waggle.plugin import Plugin, get_timestamp
import logging
import os
import base64
from urllib.parse import urlparse
import subprocess
import time


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
    try:
        subprocess.check_output([
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-i", image_uri,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-update",
            "1",
            "-y",
            "output.jpg",
        ], stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        stderr_lines = e.stderr.splitlines()
        last_stderr_line = ""
        if len(stderr_lines) > 0:
            last_stderr_line = stderr_lines[-1]
        # Raise a more informative exception, if ffmpeg provides its own error.
        if "error" in last_stderr_line.lower():
            raise RuntimeError(f"Error during RTSP fetch: {last_stderr_line}")
        # Fallback to reraising original exception.
        raise
    return Path("output.jpg").read_bytes()


def get_image_data_file(image_uri: str) -> bytes:
    logging.info("Fetching image from file %s", image_uri)
    return Path(image_uri).read_bytes()


def guess_image_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    return ""


def run(plugin: Plugin, host: str, model: str, prompt: str, images: list[Path], no_tools: bool, max_steps: int):
    logging.info("Running: model=%r and prompt=%r", model, prompt)

    client = ollama.Client(host=host)

    logging.info("Ensuring model %r has been pulled.", model)
    client.pull(model)

    for image in images:
        logging.info("Processing image: %s", image)

        state = {}

        state["image_timestamp"] = get_timestamp()
        raw_image_data = get_image_data(image)

        image_type = guess_image_type(raw_image_data)

        if not image_type:
            logging.info("Unknown image type for %s", image)
            continue

        state["image_data"] = raw_image_data
        state["image_type"] = image_type

        encoded_image_data = base64.b64encode(raw_image_data).decode()

        def upload_image(reason: str) -> str:
            """Upload the image currently being processed along with a reason for uploading the image.

            Args:
                reason: The reason for uploading the image.

            Returns:
                Confirmation on whether image was queued for upload.
            """
            filename = f"upload.{state['image_type']}"
            Path(filename).write_bytes(state['image_data'])
            plugin.upload_file(filename, timestamp=state["image_timestamp"])
            plugin.publish("upload_reason", reason, timestamp=state["image_timestamp"])
            return "Successfully queued image for upload."

        messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image_data],
                },
            ]

        if no_tools:
            logging.info("Tools are disabled.")
            tools = None
        else:
            tools = [upload_image]

        # Init metrics.
        chat_start_time = time.monotonic_ns()
        load_duration_total = 0
        prompt_eval_count_total = 0
        prompt_eval_duration_total = 0
        eval_count_total = 0
        eval_duration_total = 0
        tool_calls_total = 0

        step = 1

        while True:
            if step > max_steps:
                raise RuntimeError("Chat exceeded maximum number of steps.")
            step += 1

            logging.info("Starting chat")

            response = client.chat(
                model=model,
                messages=messages,
                tools=tools,
            )

            logging.info("Got chat response")

            messages.append(response.message)

            # Update token and duration metrics, if they exist.
            load_duration_total += response.load_duration or 0
            prompt_eval_count_total += response.prompt_eval_count or 0
            prompt_eval_duration_total += response.prompt_eval_duration or 0
            eval_count_total += response.eval_count or 0
            eval_duration_total += response.eval_duration or 0

            # Check tool calls.
            tool_calls = response.message.tool_calls

            # Stop if no more tools are being called.
            if not tool_calls:
                break

            for call in tool_calls:
                logging.info("Calling tool %s", call.function.name)
                tool_calls_total += 1
                if call.function.name == "upload_image":
                    result = upload_image(**call.function.arguments)
                else:
                    result = "Unknown tool"
                messages.append({
                    "role": "tool",
                    "tool_name": call.function.name,
                    "content": str(result),
                })

            logging.info("Chat loop is done")

        chat_stop_time = time.monotonic_ns()
        chat_duration_total = chat_stop_time - chat_start_time

        # Build output data.
        response = messages[-1]

        output = {
            "chat_duration_total": chat_duration_total / 1e9,
            "load_duration_total": load_duration_total / 1e9,
            "prompt_eval_count_total": prompt_eval_count_total,
            # convert from nanoseconds to seconds
            "prompt_eval_duration_total": prompt_eval_duration_total / 1e9,
            "eval_count_total": eval_count_total,
            # convert from nanoseconds to seconds
            "eval_duration_total": eval_duration_total / 1e9,
            "tool_calls_total": tool_calls_total,
            "model": model,
            "output": response.content,
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
    parser.add_argument("--no-tools", action="store_true", help="do not allow tool calling")
    parser.add_argument("--max-steps", type=int, default=10, help="maximum number of steps agent is allowed to take")
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
            no_tools=args.no_tools,
            max_steps=args.max_steps,
        )
