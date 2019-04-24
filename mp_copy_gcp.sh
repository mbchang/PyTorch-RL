#!/bin/bash
#cd ../../cloud/
#bash copy_gcp.sh $1
#cd -
gcloud compute scp --recurse --zone "us-east1-b" run_ant.py custom_envs core examples infra models utils debug.sh cloud_setup.sh michaelchang@instance-$1:/home/michaelchang/mp/
