from oak.run_pipeline import run_pipeline

# This is a demostration script
#
# To run it, you need to have the OAK device plugged into your device
# You also need to install our "oak" environment first
# You can find an in-depth explanation in the README
#
# Once running, the script:
# 1. Loads the images recorded by the OAK device
# 2. Performs body and face detection
# 3. Computes the scores related to stress, activity and breath
# 4. Shows the scores on real-time

if __name__ == "__main__":
    run_pipeline(video_path=None, post_server=False)
