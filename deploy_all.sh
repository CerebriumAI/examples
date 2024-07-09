#!/bin/bash

# Get a list of all directories in the current folder
dirs=($(ls -d */))

for dir in "${dirs[@]}"
do
  (
    cd "$dir" || exit
    echo "Running cerebrium deploy -y in $dir"

    # Run the deployment command
    if ! cerebrium deploy -y; then
      while true; do
        read -p "Deployment in $dir failed. Do you want to try again? (y/n): " choice
        case "$choice" in
          y|Y )
            if cerebrium deploy -y; then
              echo "Deployment in $dir succeeded."
              break
            else
              echo "Deployment in $dir failed again."
            fi
            ;;
          n|N )
            echo "Moving on to the next directory."
            break
            ;;
          * )
            echo "Please answer y or n."
            ;;
        esac
      done
    else
      echo "Deployment in $dir succeeded."
    fi
  )
done

wait
echo "All deployments finished."