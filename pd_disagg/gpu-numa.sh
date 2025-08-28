#!/bin/bash
# Usage: ./gpu_to_numa_cores.sh <GPU_ID>

GPU_ID=$1
if [[ -z "$GPU_ID" ]]; then
    echo "Usage: $0 <GPU_ID>"
    exit 1
fi

# Get PCI bus ID from nvidia-smi
GPU_PCI=$(nvidia-smi -i "$GPU_ID" --query-gpu=pci.bus_id --format=csv,noheader)
if [[ -z "$GPU_PCI" ]]; then
    echo "GPU $GPU_ID not found"
    exit 1
fi

# Extract only the last three segments (bus:device.function) to match sysfs
# Example: 000000:0E:07.0 -> 0000:07:00.0
PCI_SEGMENTS=$(echo "$GPU_PCI" | awk -F: '{printf "0000:%02x:%02x.%d\n", strtonum("0x"$2), strtonum("0x"$3), substr($4,1)}')

SYSFS_PATH="/sys/bus/pci/devices/$PCI_SEGMENTS"

if [[ ! -d "$SYSFS_PATH" ]]; then
    echo "Sysfs path $SYSFS_PATH not found for GPU $GPU_ID"
    exit 1
fi

NUMA_NODE=$(cat "$SYSFS_PATH/numa_node")
if [[ -z "$NUMA_NODE" || "$NUMA_NODE" == "-1" ]]; then
    echo "Could not determine NUMA node for GPU $GPU_ID"
    exit 1
fi

CPU_LIST=$(cat /sys/devices/system/node/node"$NUMA_NODE"/cpulist)
echo "$CPU_LIST"

