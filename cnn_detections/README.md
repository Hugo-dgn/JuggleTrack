# Juggling Motion Analysis Tools

This folder provides a suite of tools to analyze juggling videos. Using a deep learning model, it tracks the positions of balls and hands frame-by-frame. This data is then used to generate either a final, annotated video for visualization or a structured list of all catch events for further data analysis.

The system is split into two main scripts, each with a distinct purpose.

## Prerequisites

You need the pre-trained grid model (`grid_model_submovavg_64x64.h5` or similar) placed in a `../grid_models/` directory relative to the scripts.

---

## `main_analyzer.py` - Event Detection (Production)

This is the primary analysis script. Its sole purpose is to process a video and identify every instance of a ball being caught by a hand. It does not produce any video output, only a structured data report printed to the console.

### Inputs

*   `--video` (**required**): The file path to the input juggling video (e.g., `my_video.mp4`).
*   `--model` (optional): The file path to the pre-trained grid model. Defaults to `../grid_models/grid_model_submovavg_64x64.h5`.
*   `--n_balls` (optional): The number of balls in the video. Defaults to `3`.

### Output

The script prints a chronological list of all detected catch events to the terminal.

**Example Output from CLI:**
```
--- CATCH EVENT REPORT ---
Frame 58    | Ball 1 was caught by the left hand.
Frame 71    | Ball 0 was caught by the right hand.
Frame 85    | Ball 2 was caught by the left hand.
...
--------------------------
```

### Usage (from the Command Line)

To run the analysis on your video, use the following command in your terminal:

```bash
python main_analyzer.py --video <path_to_your_video.mp4> --model <path_to_your_model> --n_balls <number_of_balls>
```

### Usage (Integrating into Other Python Scripts)

The script is designed to be easily integrated into other Python pipelines. The core logic is encapsulated in the `find_catch_events` function.

To use it, import the function from the `main_analyzer` file.

*   **Function to call:** `find_catch_events(video_path, model_path, n_balls)`
*   **Return Value:** It returns a list of dictionaries. Each dictionary represents a single catch event and has the following structure:
    ```python
    {
        "catch_frame": <integer>,
        "ball_id": <integer>,
        "hand_id": <string, "left" or "right">
    }
    ```

---

## `create_visuals.py` - Annotated Video Generation (Testing & Debugging)

This script is for debugging and visualization. It runs the full analysis pipeline and uses the results to generate a new video file with detailed annotations. It helps you visually confirm the performance of the tracker and the state analysis.

### Inputs

*   `--video` (**required**): The file path to the input juggling video.
*   `--model` (optional): The file path to the pre-trained grid model. Defaults to `../grid_models/grid_model_submovavg_64x64.h5`.
*   `--output` (optional): The file path where the output annotated video will be saved. Defaults to `output_visuals.mp4`.
*   `--n_balls` (optional): The number of balls in the video. Defaults to `3`.

### Output

The script generates a new `.mp4` video file. In this video:
*   Each ball has a unique, persistent color (blue, yellow, etc.) and an ID number.
*   The border of the ball changes color to indicate its analyzed state:
    *   **White Border:** The ball is in the air (`in_flight`).
    *   **Red Border:** The ball is in the left hand.
    *   **Green Border:** The ball is in the right hand.
*   Hands are marked with red (left) and green (right) lines.

### Usage (from the Command Line)

To generate the annotated video, use the following command:

```bash
python create_visuals.py --video <path_to_your_video.mp4> --model <path_to_your_model> --output <desired_output_name.mp4> --n_balls <nbr_of_balls>
```