#!/bin/bash

# This patches the behavior of Inductor code cache to ignore CPU affinity so all CPUs can be used for parallel compilation.

# set the substring to be replaced and the new string
substring="            global _watchdog_thread"
# newstring='            os.system("taskset -p 0xff %d" % os.getpid())'
# https://github.com/cresset-template/cresset/issues/96
newstring='            os.system("taskset -c -p 0-95 %d >/dev/null 2>&1" % os.getpid())'

# loop through each line of the file
while IFS= read -r line; do
    # check if the line starts with the substring
    if [[ "$line" == "$substring"* ]]; then
        # insert the new string before the old substring
        printf '%s\n' "$newstring" >> /tmp/patched_codecache.py
    fi
    # print the updated line to a temporary file
    printf '%s\n' "$line" >> /tmp/patched_codecache.py
done < /opt/conda/lib/python3.10/site-packages/torch/_inductor/codecache.py

mv /tmp/patched_codecache.py /opt/conda/lib/python3.10/site-packages/torch/_inductor/codecache.py