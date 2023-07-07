#!/bin/bash
docker run --rm -it \
--user $(id -u):$(id -g) \
--volume "$(pwd):/workspace" \
 kubruntudev_fixed:latest \
bash
