#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# Uncomment the following lines to launch the main app
dt-exec roslaunch led_emitter led_emitter_node.launch veh:="$VEHICLE_NAME"
dt-exec roslaunch duckietown_demos deadreckoning.launch
dt-exec roslaunch apriltag_detector apriltag_detector.launch veh:="$VEHICLE_NAME"

# Uncomment the following line to launch the augmented reality basics demo
# dt-exec roslaunch augmented_reality_basics augmented_reality_basics.launch map_file:=calibration_pattern veh:="$VEHICLE_NAME"

# Uncomment the following line to launch the augmented reality apriltag demo
# dt-exec roslaunch augmented_reality_apriltag augmented_reality_apriltag.launch veh:="$VEHICLE_NAME"

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
