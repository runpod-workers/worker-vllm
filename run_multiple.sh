#!/bin/bash

# Number of concurrent requests
num_instances=40

# Start time
start_time=$(date +%s)

# Function to run the command
run_command() {
    curl -v --request POST \
         --url https://api.runpod.ai/v2/u4f0txz9dssz6q/runsync \
         --header 'accept: application/json' \
         --header 'authorization: UGXLPIBFYJXUCKOBNR8CCJ3KP8GWJNI97F91QT6Y' \
         --header 'content-type: application/json' \
         --data '
    {
      "input": {
        "prompt": "Who is the president of the United States?",
        "sampling_params": {
          "max_tokens": "300",
          "ignore_eos": true
        }
      }
    }
    '
}

# Run instances in the background
for ((i = 1; i <= num_instances; i++)); do
    run_command &
done

# Wait for all instances to finish
wait

# End time
end_time=$(date +%s)

# Calculate total time
total_time=$((end_time - start_time))

echo "Total time taken for $num_instances instances (1k tokens each): $total_time seconds"
